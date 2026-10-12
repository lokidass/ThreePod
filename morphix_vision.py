# morphix_vision.py
# See the print: screenshot the live scene, judge whether construction is
# going well. The geometric validator stays the hard gate; vision is an
# HONEST ADVISOR — it reports GOOD / DRIFTING / BROKEN per inspection, and
# two consecutive BROKEN inspections raise a visible warning suggesting the
# user press Stop. Vision never auto-fails a run on its own: a vision model
# must never be the sole authority over geometry correctness.
#
# Screenshot: headless Chromium (Playwright) over the run's scene.html.
# Judge: any Ollama vision model (moondream preferred, auto-detected).

import base64
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from morphix_config import ollama_generate
from morphix_image import available_vision_model

VISION_TIMEOUT = int(os.environ.get("MORPHIX_VISION_TIMEOUT", "150"))

JUDGE_SYSTEM = (
    "You are a 3D-print quality inspector watching a layer-by-layer print.\n"
    "Judge ONLY the printed geometry against the stated goal and progress.\n"
    "Ignore image graininess, lighting, shadows and the background grid: "
    "screenshots are plain software renders.\n"
    "Start your reply with exactly one word: GOOD, DRIFTING or BROKEN. "
    "If that is hard, start the first line with exactly one letter and "
    "nothing else: A means going well, B means drifting, C means broken. "
    "Then add one short sentence.\n"
    "GOOD = recognizably on track. DRIFTING = suspicious placement or shape "
    "but recoverable. BROKEN = empty viewport, error texture, or something "
    "entirely unrelated to the goal."
)


def screenshot_scene(url: str, out_png: str, width: int = 640,
                     height: int = 480, wait_ms: int = 2500) -> str:
    """Render url headlessly and save a PNG. Raises on failure."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width,
                                              "height": height})
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(wait_ms)
            page.screenshot(path=out_png)
        finally:
            browser.close()
    return out_png


def judge_print(png_path: str, goal: str, committed: int, layers: int,
                model: str = "") -> dict:
    """Ask the vision model whether the print is going well. Always returns
    a dict; available=False when no vision model is installed."""
    model = model or available_vision_model()
    if not model:
        return {"available": False, "verdict": "UNKNOWN",
                "note": "no vision model installed"}
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    prompt = (f"GOAL: {goal}\nPrinted so far: {committed} deposits "
              f"across {layers} layer(s).\n"
              f"Is this print going well? Start with GOOD, DRIFTING or BROKEN.")
    try:
        raw = ollama_generate(model, JUDGE_SYSTEM, prompt, images=[b64],
                              timeout=VISION_TIMEOUT)
    except Exception as e:
        return {"available": True, "model": model, "verdict": "UNKNOWN",
                "note": f"vision call failed: {e}"}
    return {"available": True, "model": model,
            "verdict": _parse_verdict(raw or ""),
            "note": _parse_note(raw or "")}


_VERDICT_RE = re.compile(r"\b(GOOD|DRIFTING|BROKEN)\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"^\s*\(?([ABC])\b[.).:]?\s*$", re.IGNORECASE)

# Fallback sentiment cues for models that describe instead of judging.
# Phrases, not bare words ("floating" alone is descriptive, not damning).
_POSITIVE_CUES = ("going well", "looks good", "looking good", "on track",
                  "good progress", "well printed", "high-quality",
                  "high quality", "great", "matches the goal", "as expected",
                  "correct shape", "coming along")
_NEGATIVE_CUES = ("not very well", "not good", "bad print", "poor quality",
                  "poor", "broken", "wrong", "empty viewport", "empty",
                  "missing", "fail", "unrelated", "nothing there",
                  "incorrect", "mess")


def _cue_verdict(text: str) -> str:
    """Majority vote over cue phrases. Ties and silence stay UNKNOWN."""
    low = text.lower()
    pos = sum(1 for c in _POSITIVE_CUES if c in low)
    neg = sum(1 for c in _NEGATIVE_CUES if c in low)
    if pos > neg:
        return "GOOD"
    if neg > pos:
        return "BROKEN"
    return "UNKNOWN"


def _parse_verdict(raw: str) -> str:
    """First token wins when valid (the instructed format); else a first
    line holding only the letter A/B/C; else the first standalone mention
    anywhere in the ramble."""
    first = raw.strip().split(None, 1)
    if first:
        tok = first[0]
        core = tok.strip(" ():.").upper()
        if core in ("GOOD", "DRIFTING", "BROKEN"):
            return core
        if len(core) == 1 and core in "ABC" and (
                tok.startswith("(") or len(tok) > 1):
            # "(b) ..." / "B. ..." style enumerated verdict. A bare "A"
            # is rejected (usually the article, as in "A yellow box").
            return {"A": "GOOD", "B": "DRIFTING", "C": "BROKEN"}[core]
    line0 = (raw.strip().split("\n")[0] if raw.strip() else "")
    m = _LETTER_RE.match(line0)
    if m:
        return {"A": "GOOD", "B": "DRIFTING", "C": "BROKEN"}[m.group(1).upper()]
    m = _VERDICT_RE.search(raw)
    if m:
        v = m.group(1).upper()
        prior = [w.strip(".,!?") for w in raw[:m.start()].lower().split()[-3:]]
        if any(w in ("not", "never", "no", "n't") or w.endswith("n't")
               for w in prior):
            # "not very good" / "isn't broken": negate the verdict
            return {"GOOD": "BROKEN", "BROKEN": "GOOD"}.get(v, "UNKNOWN")
        return v
    return _cue_verdict(raw)


def _parse_note(raw: str) -> str:
    """Sentence after the verdict token, else the first line."""
    m = _VERDICT_RE.search(raw)
    if m:
        rest = raw[m.end():].lstrip(" :,-\n")
        line = rest.split("\n")[0].strip()
        if line:
            return line[:200]
    for line in raw.split("\n"):
        if line.strip():
            return line.strip()[:200]
    return "(no explanation)"


def track_verdict(broken_streak: int, verdict: str) -> tuple:
    """Update the consecutive-BROKEN counter. Returns (new_streak, warn)."""
    if verdict == "BROKEN":
        broken_streak += 1
        if broken_streak >= 2:
            return broken_streak, (
                "vision inspector: BROKEN twice in a row — the print looks "
                "wrong. Consider pressing Stop; committed layers are kept.")
        return broken_streak, ""
    return 0, ""
