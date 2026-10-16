# morphix_image_vision.py
# Vision adapter for image -> 3D: turns whatever a vision model says into a
# NORMALIZED, validated observation before any category decision is made.
#
# Why this exists: small local vision models rarely obey a JSON schema.
# moondream (the default here) answers "Identify the main object. JSON only."
# with `ids = [0.39, 0.6, ...]`, short questions with "" or a bare "urn", and
# runs away into "car, car, car, ..." loops — while answering natural
# questions ("What is the main object in this image?") accurately. The old
# path treated those failures as category words and built an unrelated
# primitive. Here every reply is classified first:
#
#   json        structured reply (category field)          -> strong
#   sentence    a sentence whose grammatical SUBJECT names  -> strong
#               the object ("The main object ... is a sports car")
#   bare_word   a lone word ("urn")                         -> weak only
#   degenerate  repetition loop ("car, car, car, ...")      -> ignored
#   malformed   code/garbage where an answer was expected   -> ignored
#   empty       no text                                     -> ignored
#
# The category comes ONLY from json fields or a sentence's subject noun
# phrase, resolved through a synonym lexicon (automobile, SUV, Lamborghini
# -> car). It is NEVER found by searching for category words inside a free
# text parts description. Decisions need agreement across independent
# questions (or one strong answer plus a geometric prior from the measured
# silhouette); failures are reported as distinct kinds:
#
#   unavailable     the model could not be reached
#   malformed       the model answered, but nothing usable
#   low_confidence  only weak (bare word) answers
#   ambiguous       strong answers disagree
#
# Questions are bounded (MAX_QUESTIONS); there is no open-ended retry.

import re
from collections import Counter

MAX_QUESTIONS = 5

# Families group synonyms and decide which shape builder applies.
# category -> family. Categories without a dedicated builder still get a
# family-level parametric builder or the measured-silhouette builder.
FAMILY = {
    "car": "vehicle", "truck": "vehicle", "van": "vehicle",
    "bus": "vehicle",
    "table": "table", "desk": "table",
    "chair": "chair", "stool": "stool", "bench": "table",
    "house": "house", "barn": "house", "shed": "house", "cabin": "house",
    "cat": "quadruped", "dog": "quadruped", "horse": "quadruped",
    "cow": "quadruped", "sheep": "quadruped", "fox": "quadruped",
    "pig": "quadruped", "goat": "quadruped", "deer": "quadruped",
    "lion": "quadruped", "tiger": "quadruped", "wolf": "quadruped",
    "animal": "quadruped",
    "vase": "vessel", "urn": "vessel", "bottle": "vessel", "jar": "vessel",
    "cup": "vessel", "mug": "vessel", "bowl": "vessel", "pot": "vessel",
    "glass": "vessel", "lamp": "vessel",
}

SYNONYMS = {
    # vehicles
    "car": "car", "cars": "car", "automobile": "car", "vehicle": "car",
    "sedan": "car", "suv": "car", "coupe": "car", "hatchback": "car",
    "convertible": "car", "roadster": "car", "supercar": "car",
    "hypercar": "car", "racecar": "car", "limousine": "car", "taxi": "car",
    "wagon": "car", "minivan": "car", "crossover": "car", "jeep": "car",
    "truck": "truck", "pickup": "truck", "lorry": "truck", "van": "van",
    "bus": "bus",
    # furniture
    "table": "table", "desk": "table", "workbench": "table",
    "chair": "chair", "armchair": "chair", "stool": "stool",
    "barstool": "stool", "bench": "bench",
    # buildings
    "house": "house", "home": "house", "cottage": "house",
    "bungalow": "house", "villa": "house", "hut": "house",
    "building": "house", "barn": "barn", "shed": "shed", "cabin": "cabin",
    # animals
    "cat": "cat", "kitten": "cat", "kitty": "cat", "dog": "dog",
    "puppy": "dog", "horse": "horse", "pony": "horse", "cow": "cow",
    "sheep": "sheep", "lamb": "sheep", "fox": "fox", "pig": "pig",
    "goat": "goat", "deer": "deer", "lion": "lion", "tiger": "tiger",
    "wolf": "wolf", "animal": "animal",
    # vessels
    "vase": "vase", "urn": "urn", "bottle": "bottle", "jar": "jar",
    "cup": "cup", "mug": "mug", "bowl": "bowl", "pot": "pot",
    "lamp": "lamp",
}

# Multi-word subject terms checked before single words.
PHRASES = {"sports car": "car", "race car": "car", "police car": "car",
           "pickup truck": "truck", "coffee table": "table",
           "dining table": "table", "office chair": "chair",
           "rocking chair": "chair", "bar stool": "stool"}

# Vehicle makes: a sentence subject "a black Lamborghini Huracan" is a car.
CAR_MAKES = {
    "lamborghini", "ferrari", "porsche", "bmw", "audi", "mercedes",
    "benz", "tesla", "toyota", "honda", "ford", "chevrolet", "chevy",
    "nissan", "volkswagen", "vw", "mclaren", "bugatti", "jaguar", "lexus",
    "hyundai", "kia", "mazda", "subaru", "volvo", "bentley", "maserati",
    "aston", "dodge", "cadillac", "buick", "fiat", "peugeot", "renault",
    "koenigsegg", "pagani", "lotus", "alfa", "acura", "infiniti",
    "mitsubishi", "suzuki", "skoda", "seat", "opel", "citroen", "rivian",
    "lucid", "polestar", "genesis", "lincoln", "chrysler", "ram", "gmc"}

# Words that are never the photographed object itself.
NOT_SUBJECT = {
    "image", "picture", "photo", "photograph", "rendering", "render",
    "scene", "view", "background", "foreground", "color", "colour",
    "side", "front", "back", "rear", "top", "bottom", "left", "right",
    "middle", "center", "centre", "floor", "ground", "road", "driveway",
    "platform", "wall", "sky", "light", "shadow", "surface", "area",
    "object", "item", "thing", "shape", "model", "design", "style",
    "sleek", "black", "white", "red", "blue", "green", "yellow", "dark",
    "light", "large", "small", "big", "modern", "classic", "luxury",
    "shiny", "silver", "gray", "grey", "orange", "brown", "pink",
    "purple", "gold", "golden", "wooden", "metal", "metallic", "new",
    "old", "beautiful", "single", "main", "close", "up", "3d", "digital",
    "computer", "generated", "of", "and", "or", "very", "high", "low",
    "sporty", "elegant", "simple", "plain", "cute", "little", "tall",
    "short", "long", "wide", "round", "square", "rectangular", "parked",
}

PART_LEXICON = {
    "vehicle": {"wheel", "wheels", "tire", "tires", "tyre", "tyres",
                "hood", "bonnet", "door", "doors", "window", "windows",
                "windshield", "windscreen", "bumper", "headlight",
                "headlights", "taillight", "taillights", "roof", "trunk",
                "mirror", "mirrors", "grille", "grill", "spoiler", "rim",
                "rims", "fender", "body"},
    "table": {"leg", "legs", "top", "tabletop", "surface", "apron",
              "drawer"},
    "chair": {"seat", "back", "backrest", "leg", "legs", "armrest",
              "armrests", "cushion"},
    "stool": {"seat", "leg", "legs", "footrest"},
    "house": {"roof", "door", "doors", "window", "windows", "wall",
              "walls", "chimney", "porch", "steps", "garage"},
    "quadruped": {"head", "ear", "ears", "tail", "leg", "legs", "paw",
                  "paws", "body", "eye", "eyes", "nose", "fur", "whiskers",
                  "mane", "snout", "hooves"},
    "vessel": {"neck", "rim", "handle", "handles", "base", "foot", "body",
               "lid", "spout", "mouth", "shoulder"},
}

COLOR_WORDS = {"red", "blue", "green", "yellow", "black", "white", "brown",
               "gray", "grey", "orange", "cyan", "teal", "silver", "gold",
               "pink", "purple", "beige", "wooden", "dark", "light"}

ENV_WORDS = {"platform", "floor", "road", "driveway", "street", "grass",
             "table", "pedestal", "turntable", "stage", "wall", "studio",
             "background", "garage", "parking", "spotlight", "shadow"}

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6}

# Geometric priors (measured subject bbox aspect, width/height) per family.
# Deliberately wide: a prior only vetoes a single uncorroborated answer
# that contradicts the photo's silhouette.
ASPECT_PRIOR = {"vehicle": (0.8, 5.0), "table": (0.5, 5.0),
                "chair": (0.25, 1.8), "stool": (0.3, 1.8),
                "house": (0.4, 3.5), "quadruped": (0.5, 4.0),
                "vessel": (0.15, 2.0)}

# Model profiles: JSON prompting is only attempted on models that follow it.
JSON_CAPABLE = ("llava", "qwen2.5vl", "qwen2-vl", "llama3.2-vision",
                "minicpm-v", "gemma3", "granite3.2-vision")

JSON_SYSTEM = (
    "You are a vision analyst. Identify the single main physical object in "
    "the photograph. Reply with ONLY a JSON object: {\"category\": "
    "\"<common noun>\", \"confidence\": <0..1>, \"parts\": [...], "
    "\"view\": \"side|front|rear|angle|top\"}")

SUBJECT_QUESTIONS = (
    "What is the main object in this image?",
    "Describe this image.",
    "What kind of object is shown in the middle of this picture?",
    "Name the object in the center of this photo and describe its shape.",
)


class VisionFailure(RuntimeError):
    """The vision stage could not produce a trustworthy subject. ``kind`` is
    one of unavailable | malformed | low_confidence | ambiguous; ``info``
    carries every reply and the evidence for diagnostics."""

    def __init__(self, kind: str, message: str, info: dict = None):
        super().__init__(message)
        self.kind = kind
        self.info = dict(info or {})


# ------------------------------------------------------------ reply parsing
_WORD = re.compile(r"[a-z][a-z0-9'-]*")


def _words(text: str) -> list:
    return _WORD.findall((text or "").lower())


def classify_reply(raw: str, expect_json: bool = False) -> str:
    """Shape of one reply: json | sentence | bare_word | degenerate |
    malformed | empty."""
    t = (raw or "").strip()
    if not t:
        return "empty"
    if expect_json:
        if _json_object(t) is not None:
            return "json"
    words = _words(t)
    if len(words) >= 12:
        top = Counter(words).most_common(1)[0][1]
        if top / len(words) >= 0.4:
            return "degenerate"
    if re.search(r"^\s*[a-z_]+\s*=\s*[\[{(]", t, re.I) \
            or re.fullmatch(r"[\s\[\]{}(),.:;0-9-]+", t):
        return "malformed"
    if _json_object(t) is not None:
        return "json"
    if len(words) == 1 and re.fullmatch(r"[A-Za-z][A-Za-z-]{1,30}[.!]?", t):
        return "bare_word"
    if len(words) >= 3:
        return "sentence"
    return "malformed"


def _json_object(t: str):
    import json
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return v if isinstance(v, dict) else None


_CUES = (
    re.compile(r"\bmain (?:object|subject|item|focus)\b[^.]*?\bis\s+"
               r"(?:a|an|the)\s+(?P<np>[^.,;:\n!?]+)", re.I),
    re.compile(r"\b(?:image|picture|photo|photograph|rendering|render|scene"
               r"|illustration)\s+(?:shows|presents|features|depicts|"
               r"displays|contains|captures|showcases|is of|portrays)\s+"
               r"(?:a|an|the|two|three|four)?\s*(?P<np>[^.,;:\n!?]+)", re.I),
    re.compile(r"^\s*(?:this is|it is|it's|there is|i see|shown is)\s+"
               r"(?:a|an|the)\s+(?P<np>[^.,;:\n!?]+)", re.I),
    re.compile(r"^\s*(?:the object is|the object shown is|the object in the "
               r"(?:center|centre|middle)[^.]*? is)\s+(?:a|an|the)\s+"
               r"(?P<np>[^.,;:\n!?]+)", re.I),
    re.compile(r"^\s*(?:a|an|the)\s+(?P<np>[^.,;:\n!?]+)", re.I),
)

_NP_STOP = re.compile(
    r"\b(?:on|in|at|with|parked|sitting|sits|standing|stands|placed|against|"
    r"near|under|over|that|which|who|is|are|was|facing|resting|positioned|"
    r"set|lying|located|from|by|and|as|while|for|to)\b", re.I)


_DEPICTION = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:(?:3d|digital|computer[- ]generated|realistic|"
    r"detailed|close[- ]up|black and white|stylized)\s+)*"
    r"(?:rendering|render|image|picture|photo|photograph|drawing|model|"
    r"illustration|sketch|painting|scan|view|close-up|closeup)\s+of\s+"
    r"(?:(?:a|an|the|two|three)\s+)?", re.I)


def subject_phrase(text: str):
    """The grammatical subject noun phrase of a descriptive sentence, or
    None. Only the phrase introduced by a subject cue is returned — later
    mentions ("... next to a table", "wheels, doors ...") never count."""
    for line in re.split(r"(?<=[.!?])\s+|\n", (text or "").strip()):
        for cue in _CUES:
            m = cue.search(line)
            if m:
                # "3D rendering of a skull" / "a photo of a red car": the
                # subject is what the depiction is OF, not the depiction
                raw = _DEPICTION.sub("", m.group("np").strip())
                np_ = _NP_STOP.split(raw)[0].strip()
                if np_:
                    return np_
        break  # only the first sentence carries the subject
    return None


def resolve_category(phrase: str):
    """(category, term, family) for a subject noun phrase. Unknown head nouns
    return (head, head, None) — a genuine identification without a
    dedicated builder, never a forced template."""
    p = " ".join(_words(phrase))
    for ph, cat in PHRASES.items():
        if ph in p:
            return cat, ph, FAMILY.get(cat)
    words = [w for w in _words(phrase)]
    for w in reversed(words):
        base = w[:-1] if w.endswith("s") and w[:-1] in SYNONYMS else w
        if base in SYNONYMS:
            cat = SYNONYMS[base]
            return cat, w, FAMILY.get(cat)
    for w in words:
        if w in CAR_MAKES:
            return "car", w, "vehicle"
    heads = [w for w in words if w not in NOT_SUBJECT and len(w) > 2
             and not w.isdigit()]
    if heads:
        h = heads[-1]
        return h, h, None
    return None, None, None


def extract_subject(raw: str, kind: str):
    """Evidence dict from one classified reply, or None."""
    if kind == "json":
        obj = _json_object(raw) or {}
        val = str(obj.get("category") or "").strip()
        if not val:
            return None
        cat, term, fam = resolve_category(val)
        if not cat:
            return None
        try:
            conf = float(obj.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        return {"category": cat, "family": fam, "term": term,
                "phrase": val, "strength": "strong" if conf >= 0.4
                else "weak", "kind": "json", "confidence": conf,
                "view": obj.get("view"), "parts": obj.get("parts")}
    if kind == "sentence":
        np_ = subject_phrase(raw)
        if not np_:
            return None
        cat, term, fam = resolve_category(np_)
        if not cat:
            return None
        return {"category": cat, "family": fam, "term": term,
                "phrase": np_, "strength": "strong", "kind": "sentence"}
    if kind == "bare_word":
        w = _words(raw)[0]
        cat = SYNONYMS.get(w, w)
        return {"category": cat, "family": FAMILY.get(cat), "term": w,
                "phrase": w, "strength": "weak", "kind": "bare_word"}
    return None


def parse_view(text: str):
    t = " ".join(_words(text))
    if not t:
        return None
    has = {k: bool(re.search(p, t)) for k, p in (
        ("angle", r"\b(angle|angled|diagonal|three quarter|three-quarter|"
                  r"corner|perspective)\b"),
        ("side", r"\b(side|profile|sideways)\b"),
        ("front", r"\b(front|head on|head-on|frontal)\b"),
        ("rear", r"\b(back|rear|behind)\b"),
        ("top", r"\b(above|top down|overhead|bird)\b"))}
    if has["angle"] or (has["front"] and has["side"]) \
            or (has["rear"] and has["side"]):
        return "angle_rear" if has["rear"] and not has["front"] else "angle"
    for k in ("side", "front", "rear", "top"):
        if has[k]:
            return k
    return None


def parse_facing(text: str):
    """Image-space direction the subject's front points ("facing towards the
    right side of the image") -> right | left | camera | None. Advisory:
    small models contradict themselves, so the silhouette fit decides."""
    t = " ".join(_words(text))
    m = re.search(r"facing (?:towards |toward |to )?(?:the )?"
                  r"(right|left|camera|viewer|us|forward)", t)
    if not m:
        return None
    v = m.group(1)
    return v if v in ("right", "left") else "camera"


# Approximate sRGB for colour words (used only to CORROBORATE measured
# regions — never to paint geometry).
NAMED_COLORS = {"red": "0xC83232", "blue": "0x3060C8", "green": "0x3C9640",
                "yellow": "0xE6C828", "black": "0x141414",
                "white": "0xF0F0F0", "brown": "0x785032", "gray": "0x808080",
                "grey": "0x808080", "orange": "0xE68C32",
                "silver": "0xB4B8BC", "gold": "0xC8A03C", "pink": "0xE68CB4",
                "purple": "0x7840A0", "beige": "0xD2C3A0", "cyan": "0x30C8C8",
                "teal": "0x2E8C8C", "wooden": "0x8C5A32"}


def co_objects(texts) -> list:
    """Other objects named next to the subject: [{"color", "noun"}] from
    "next to / beside / behind / in front of / near a (colour) noun"."""
    out = []
    pat = re.compile(r"(?:next to|beside|behind|in front of|near|"
                     r"alongside|by)\s+(?:a|an|the)\s+((?:[a-z]+\s+){0,2}?)"
                     r"([a-z]+)")
    for t in texts:
        for m in pat.finditer((t or "").lower()):
            words = m.group(1).split()
            col = next((w for w in words if w in NAMED_COLORS), None)
            out.append({"color": col, "noun": m.group(2)})
    return out


def parse_parts(text: str, family: str) -> list:
    lex = PART_LEXICON.get(family or "", set())
    found = []
    for w in _words(text):
        if w in lex and w not in found:
            found.append(w)
    return found


def parse_count(text: str, noun: str):
    m = re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:\w+\s+)?"
                  + noun, (text or "").lower())
    if not m:
        return None
    v = m.group(1)
    return int(v) if v.isdigit() else NUMBER_WORDS[v]


# ------------------------------------------------------------ the adapter
def observe(ask, model: str, aspect: float = None, emit=None,
            max_questions: int = MAX_QUESTIONS) -> dict:
    """Ask a bounded set of natural questions and decide the subject.

    ``ask(system, prompt) -> str`` performs one vision call (raises
    morphix_config.ModelUnavailableError on transport failure).
    ``aspect`` is the measured subject silhouette aspect (width/height)
    used as a geometric prior. Returns the normalized observation dict;
    raises VisionFailure with a specific kind otherwise."""
    from morphix_config import ModelUnavailableError
    emit = emit or (lambda e: None)
    replies, evidence = [], []
    json_ok = any(model.lower().startswith(p) for p in JSON_CAPABLE)
    plan = ([("json", JSON_SYSTEM, "Identify the main object. JSON only.")]
            if json_ok else [])
    plan += [("subject", "", q) for q in SUBJECT_QUESTIONS]

    def call(tag, system, prompt):
        try:
            raw = ask(system, prompt)
        except ModelUnavailableError as e:
            raise VisionFailure(
                "unavailable", f"vision model {model!r} unavailable: {e}",
                {"model": model, "replies": replies})
        kind = classify_reply(raw, expect_json=(tag == "json"))
        ev = extract_subject(raw, kind) if tag in ("json", "subject") \
            else None
        rec = {"question": prompt, "tag": tag, "raw": (raw or "")[:1500],
               "kind": kind, "evidence": ev}
        replies.append(rec)
        emit({"type": "vision_reply", "question": prompt, "kind": kind,
              "subject": (ev or {}).get("category")})
        if ev:
            evidence.append(ev)
        return raw, kind

    decision = None
    asked = 0
    for tag, system, prompt in plan:
        if asked >= max_questions:
            break
        call(tag, system, prompt)
        asked += 1
        # Never decide on the first answer alone: at least two questions.
        # Stop early only on agreement; a single answer (+ prior) is
        # accepted after the bounded question budget is spent.
        if asked >= 2:
            decision = decide(evidence, aspect)
            if decision["status"] == "ok" and decision["confidence"] >= 0.7:
                break
    if decision is None:
        decision = decide(evidence, aspect)
    if decision["status"] != "ok":
        kinds = Counter(r["kind"] for r in replies)
        if decision["status"] == "none":
            status = ("low_confidence" if any(
                e["strength"] == "weak" for e in evidence) else "malformed")
        else:
            status = decision["status"]
        raise VisionFailure(
            status,
            f"vision could not identify the subject ({status}): "
            f"{decision['reason']}; replies: "
            + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
            {"model": model, "replies": replies, "evidence": evidence,
             "decision": decision})

    cat, fam = decision["category"], decision["family"]
    noun = cat if fam != "vehicle" else "car"
    view_raw, _ = call("view", "",
                       f"Is the {noun} seen from the side, from the front, "
                       f"from the back, or at an angle?")
    parts_raw, _ = call("parts", "",
                        f"Describe the visible parts of the {noun} in this "
                        f"image.")
    describe = next((r["raw"] for r in replies
                     if r["question"] == "Describe this image."), "")
    text_all = " ".join([parts_raw or "", describe or ""])
    obs = {
        "model": model,
        "available": True,
        "status": "ok",
        "category": cat,
        "family": fam,
        "label": decision.get("phrase"),
        "confidence": decision["confidence"],
        "decided_by": decision["reason"],
        "dissent": decision.get("dissent", []),
        "viewpoint": {"label": parse_view(view_raw), "raw": view_raw,
                      "facing": parse_facing(describe)},
        "parts": parse_parts(parts_raw, fam),
        "wheel_count": parse_count(text_all, "wheels?") if fam == "vehicle"
        else None,
        "colors": [w for w in dict.fromkeys(_words(describe))
                   if w in COLOR_WORDS],
        "subject_colors": [w for w in _words(decision.get("phrase") or "")
                           if w in NAMED_COLORS],
        "co_objects": co_objects([r["raw"] for r in replies]),
        "environment_words": [w for w in dict.fromkeys(_words(describe))
                              if w in ENV_WORDS and w != cat],
        "replies": replies,
        "evidence": evidence,
    }
    return obs


def decide(evidence: list, aspect: float = None) -> dict:
    """Decide the subject from normalized evidence. Strong evidence only:
    json fields and sentence subjects. Bare words never decide."""
    strong = [e for e in evidence if e["strength"] == "strong"]
    if not strong:
        return {"status": "none",
                "reason": "no reply named the subject in a sentence or a "
                          "structured field"}
    by_family = Counter((e["family"] or e["category"]) for e in strong)
    ranked = by_family.most_common()
    top_key, top_n = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_n:
        return {"status": "ambiguous",
                "reason": "strong answers disagree: " + ", ".join(
                    f"{k} x{n}" for k, n in ranked),
                "candidates": [k for k, _ in ranked]}
    members = [e for e in strong if (e["family"] or e["category"]) == top_key]
    cat = Counter(e["category"] for e in members).most_common(1)[0][0]
    fam = members[0]["family"]
    dissent = [f"{e['kind']}: {e['phrase']!r} -> {e['category']}"
               for e in strong if (e["family"] or e["category"]) != top_key]
    prior = ASPECT_PRIOR.get(fam or "")
    fits = (aspect is None or prior is None
            or prior[0] <= aspect <= prior[1])
    phrase = members[0]["phrase"]
    if top_n >= 2 and not dissent:
        conf = 0.9 if fits else 0.6
        return {"status": "ok", "category": cat, "family": fam,
                "phrase": phrase, "confidence": conf,
                "reason": f"{top_n} independent answers agree on "
                          f"{cat!r} ({', '.join(e['phrase'] for e in members)})"
                          + ("" if fits else
                             "; silhouette outside the family prior")}
    if top_n >= 2:
        return {"status": "ok", "category": cat, "family": fam,
                "phrase": phrase, "confidence": 0.7 if fits else 0.5,
                "dissent": dissent,
                "reason": f"majority {top_n}/{len(strong)} for {cat!r}; "
                          f"dissent: {'; '.join(dissent)}"}
    if not dissent and fits and prior is not None:
        return {"status": "ok", "category": cat, "family": fam,
                "phrase": phrase, "confidence": 0.6,
                "reason": f"one answer names {cat!r} ({phrase!r}) and the "
                          f"measured silhouette aspect {aspect:.2f} fits the "
                          f"{fam} prior {prior}"}
    if dissent:
        return {"status": "ambiguous",
                "reason": f"single answers disagree: {cat!r} vs "
                          f"{'; '.join(dissent)}",
                "candidates": [k for k, _ in ranked]}
    return {"status": "none" if prior is None and fits else "low_confidence",
            "reason": f"only one answer names {cat!r}"
                      + ("" if fits else
                         f" and the silhouette aspect {aspect} contradicts "
                         f"the {fam} prior {prior}")}


def ollama_ask(path: str, model: str):
    """Default ``ask`` bound to one image through Ollama."""
    import base64
    from morphix_config import ollama_generate
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    def ask(system, prompt):
        return ollama_generate(model, system, prompt, images=[b64],
                               options={"temperature": 0.0, "num_ctx": 2048,
                                        "num_predict": 160})
    return ask
