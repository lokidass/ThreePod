# tests/test_scene_api.py
# Scene-mode HTTP API: template catalogue, deterministic scene generation
# (template / plan / prompt paths), error contracts. Uses Flask's test
# client; the thinker path is mocked (no live model).
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "RUNS_DIR", str(tmp_path))
    server.app.config["TESTING"] = True
    return server.app.test_client()


def test_template_catalogue_lists_tested_templates(client):
    r = client.get("/api/scene-templates")
    assert r.status_code == 200
    body = r.get_json()
    names = [t["name"] for t in body["templates"]]
    for want in ("fog_benchmark", "forest", "city", "water"):
        assert want in names, names
    assert body["backend"] == "webgl"
    assert all("error" not in t for t in body["templates"])


def test_generate_scene_from_template(client):
    r = client.post("/api/scenes", json={"template": "fog_benchmark",
                                         "seed": 152})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["mode"] == "scene"
    assert body["report"]["backend"] == "webgl"
    assert body["ir"]["nodes"] >= 2
    assert "window.__scene_ready" in body["scene_js"]
    run_id = body["run_id"]
    html = client.get(f"/runs/{run_id}/scene.html")
    assert html.status_code == 200
    assert "importmap" in html.get_data(as_text=True)


def test_unknown_template_is_400_not_substitution(client):
    r = client.post("/api/scenes", json={"template": "dragon_castle"})
    assert r.status_code == 400
    assert "dragon_castle" in r.get_json()["error"]


def test_generate_scene_from_client_plan(client):
    plan = {"objective": "t", "style": "s", "backend": "webgl",
            "templates": ["city"], "seed": 4, "animations": False}
    r = client.post("/api/scenes", json={"plan": plan})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["ir"]["nodes"] >= 1


def test_bad_plan_is_422(client):
    r = client.post("/api/scenes",
                    json={"plan": {"templates": ["nope"], "seed": 1}})
    assert r.status_code == 422
    r = client.post("/api/scenes", json={})
    assert r.status_code == 400


def test_generate_scene_from_prompt_uses_thinker(monkeypatch, client):
    import morphix_config
    monkeypatch.setattr(
        morphix_config, "ollama_generate",
        lambda *a, **k: ('{"objective": "o", "style": "s", '
                         '"backend": "webgl", "templates": ["water"], '
                         '"seed": 9, "animations": true}'))
    r = client.post("/api/scenes", json={"prompt": "a calm lake"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["report"]["plan"]["source"] == "thinker"
