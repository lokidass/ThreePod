# tests/visual_geometry/fixtures.py
# Deterministic visual-geometry fixtures. Plans are fixed literal data
# copied from the production canonical templates (ir_for_category), so a
# template change fails these tests loudly for human re-approval instead of
# silently moving the goalposts. Expectations (bbox, contacts, materials)
# are hand-written intent, independent of the compiler under test.
import copy

CUBE_PLAN = [
    {"action": "add_box", "width": 1.0, "height": 1.0, "depth": 1.0,
     "position": [0, 0.5, 0], "color": "0xFFAA33", "label": "cube",
     "material": "body"},
]

SPHERE_PLAN = [
    {"action": "add_sphere", "radius": 0.6, "position": [0, 0.6, 0],
     "color": "0x44CCFF", "label": "ball", "material": "accent"},
]

CHAIR_PLAN = [
    {"label": "seat", "position": [0.0, 0.51, 0.0], "material": "wood",
     "color": "0x9C6B3F", "action": "add_box", "width": 1.1, "height": 0.12,
     "depth": 1.1},
    {"label": "backrest", "position": [-0.49, 1.02, 0.0], "material": "wood",
     "color": "0x9C6B3F", "action": "add_box", "width": 0.12, "height": 0.9,
     "depth": 1.0},
    {"label": "leg_front_left", "position": [0.45, 0.225, 0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
    {"label": "leg_front_right", "position": [0.45, 0.225, -0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
    {"label": "leg_rear_left", "position": [-0.45, 0.225, 0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
    {"label": "leg_rear_right", "position": [-0.45, 0.225, -0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
]

STOOL_PLAN = [
    {"label": "seat", "position": [0.0, 0.51, 0.0], "material": "wood",
     "color": "0x9C6B3F", "action": "add_box", "width": 1.0, "height": 0.12,
     "depth": 1.0},
    {"label": "leg_front_left", "position": [0.4, 0.225, 0.4],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
    {"label": "leg_front_right", "position": [0.4, 0.225, -0.4],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
    {"label": "leg_rear_left", "position": [-0.4, 0.225, 0.4],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
    {"label": "leg_rear_right", "position": [-0.4, 0.225, -0.4],
     "material": "wood", "color": "0x9C6B3F", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.07, "height": 0.45},
]

TABLE_PLAN = [
    {"label": "leg_front_left", "position": [0.85, 0.5, 0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_box",
     "width": 0.2, "height": 1.0, "depth": 0.2},
    {"label": "leg_front_right", "position": [0.85, 0.5, -0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_box",
     "width": 0.2, "height": 1.0, "depth": 0.2},
    {"label": "leg_rear_left", "position": [-0.85, 0.5, 0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_box",
     "width": 0.2, "height": 1.0, "depth": 0.2},
    {"label": "leg_rear_right", "position": [-0.85, 0.5, -0.45],
     "material": "wood", "color": "0x9C6B3F", "action": "add_box",
     "width": 0.2, "height": 1.0, "depth": 0.2},
    {"label": "tabletop", "position": [0.0, 1.1, 0.0], "material": "wood",
     "color": "0x9C6B3F", "action": "add_box", "width": 2.0, "height": 0.2,
     "depth": 1.2},
]

CAR_PLAN = [
    {"label": "car_body", "position": [0.0, 0.8, 0.0], "material": "body",
     "color": "0xFFAA33", "action": "add_box", "width": 3.6, "height": 0.6,
     "depth": 1.8},
    {"label": "cabin", "position": [0.0, 1.375, 0.0], "material": "accent",
     "color": "0x44CCFF", "action": "add_box", "width": 1.8, "height": 0.55,
     "depth": 1.6},
    {"label": "windshield", "position": [0.9, 1.375, 0.0],
     "material": "glass", "color": "0x99DDFF", "action": "add_box",
     "width": 0.12, "height": 0.4, "depth": 1.5},
    {"label": "side_window_left", "position": [0.0, 1.375, 0.81],
     "material": "glass", "color": "0x99DDFF", "action": "add_box",
     "width": 1.3, "height": 0.3, "depth": 0.12},
    {"label": "side_window_right", "position": [0.0, 1.375, -0.81],
     "material": "glass", "color": "0x99DDFF", "action": "add_box",
     "width": 1.3, "height": 0.3, "depth": 0.12},
    {"label": "wheel_front_left", "position": [1.3, 0.25, 0.75],
     "material": "tire", "color": "0x222222", "action": "add_cylinder",
     "radiusTop": 0.35, "radiusBottom": 0.35, "height": 0.5},
    {"label": "wheel_front_right", "position": [1.3, 0.25, -0.75],
     "material": "tire", "color": "0x222222", "action": "add_cylinder",
     "radiusTop": 0.35, "radiusBottom": 0.35, "height": 0.5},
    {"label": "wheel_rear_left", "position": [-1.3, 0.25, 0.75],
     "material": "tire", "color": "0x222222", "action": "add_cylinder",
     "radiusTop": 0.35, "radiusBottom": 0.35, "height": 0.5},
    {"label": "wheel_rear_right", "position": [-1.3, 0.25, -0.75],
     "material": "tire", "color": "0x222222", "action": "add_cylinder",
     "radiusTop": 0.35, "radiusBottom": 0.35, "height": 0.5},
    {"label": "headlight_left", "position": [1.72, 0.85, 0.55],
     "material": "accent", "color": "0x44CCFF", "action": "add_box",
     "width": 0.15, "height": 0.2, "depth": 0.35},
    {"label": "headlight_right", "position": [1.72, 0.85, -0.55],
     "material": "accent", "color": "0x44CCFF", "action": "add_box",
     "width": 0.15, "height": 0.2, "depth": 0.35},
]

BOX_SPHERE_PLAN = [
    {"label": "box", "position": [0.0, 0.5, 0.0], "material": "blue",
     "color": "0x2A6FDB", "action": "add_box", "width": 1.4, "height": 1.0,
     "depth": 1.4},
    {"label": "sphere", "position": [0.0, 1.6, 0.0], "material": "accent",
     "color": "0x44CCFF", "action": "add_sphere", "radius": 0.6},
]


def _legs(prefixes=("leg_front_left", "leg_front_right", "leg_rear_left",
                     "leg_rear_right")):
    return list(prefixes)


CAT_PLAN = [
    {"label": "body", "position": [0.0, 0.75, 0.0], "material": "body",
     "color": "0xFFAA33", "action": "add_box", "width": 1.3, "height": 0.7,
     "depth": 0.7},
    {"label": "neck", "position": [0.45, 1.25, 0.0], "material": "body",
     "color": "0xFFAA33", "action": "add_cylinder", "radiusTop": 0.2,
     "radiusBottom": 0.26, "height": 0.3},
    {"label": "head", "position": [0.45, 1.76, 0.0], "material": "body",
     "color": "0xFFAA33", "action": "add_sphere", "radius": 0.36},
    {"label": "ear_left", "position": [0.45, 2.26, 0.16],
     "material": "body", "color": "0xFFAA33", "action": "add_cylinder",
     "radiusTop": 0.03, "radiusBottom": 0.12, "height": 0.28},
    {"label": "ear_right", "position": [0.45, 2.26, -0.16],
     "material": "body", "color": "0xFFAA33", "action": "add_cylinder",
     "radiusTop": 0.03, "radiusBottom": 0.12, "height": 0.28},
    {"label": "leg_front_left", "position": [0.5, 0.2, 0.25],
     "material": "body", "color": "0xFFAA33", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.1, "height": 0.4},
    {"label": "leg_front_right", "position": [0.5, 0.2, -0.25],
     "material": "body", "color": "0xFFAA33", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.1, "height": 0.4},
    {"label": "leg_rear_left", "position": [-0.5, 0.2, 0.25],
     "material": "body", "color": "0xFFAA33", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.1, "height": 0.4},
    {"label": "leg_rear_right", "position": [-0.5, 0.2, -0.25],
     "material": "body", "color": "0xFFAA33", "action": "add_cylinder",
     "radiusTop": 0.07, "radiusBottom": 0.1, "height": 0.4},
    {"label": "tail", "position": [-0.74, 0.6, 0.25], "material": "body",
     "color": "0xFFAA33", "action": "add_cylinder", "radiusTop": 0.04,
     "radiusBottom": 0.09, "height": 1.2},
]


FIXTURES = {
    # name: plan, spec, bbox ((min),(max)) incl. bead half-width 0.1,
    # contacts [(a, b)] with "ground" meaning min-y contact, materials.
    "cube": {
        "plan": CUBE_PLAN, "spec": None,
        "bbox": ([-0.6, -0.1, -0.6], [0.6, 1.1, 0.6]),
        "contacts": [("cube", "ground")],
        "materials": {"cube": "body"},
    },
    "sphere": {
        "plan": SPHERE_PLAN, "spec": None,
        "bbox": ([-0.7, -0.1, -0.7], [0.7, 1.3, 0.7]),
        "contacts": [("ball", "ground")],
        "materials": {"ball": "accent"},
    },
    "chair": {
        "plan": CHAIR_PLAN, "spec": "chair",
        "bbox": ([-0.65, -0.1, -0.65], [0.65, 1.57, 0.65]),
        "contacts": [(leg, "seat") for leg in _legs()] + [
            ("backrest", "seat")],
        "materials": {"seat": "wood", "backrest": "wood",
                      "leg_front_left": "wood", "leg_front_right": "wood",
                      "leg_rear_left": "wood", "leg_rear_right": "wood"},
    },
    "stool": {
        "plan": STOOL_PLAN, "spec": "stool",
        "bbox": ([-0.6, -0.1, -0.6], [0.6, 0.67, 0.6]),
        "contacts": [(leg, "seat") for leg in _legs()],
        "materials": {"seat": "wood",
                      "leg_front_left": "wood", "leg_front_right": "wood",
                      "leg_rear_left": "wood", "leg_rear_right": "wood"},
    },
    "table": {
        "plan": TABLE_PLAN, "spec": "table",
        "bbox": ([-1.1, -0.1, -0.7], [1.1, 1.3, 0.7]),
        "contacts": [(leg, "tabletop") for leg in _legs()],
        "materials": {"tabletop": "wood",
                      "leg_front_left": "wood", "leg_front_right": "wood",
                      "leg_rear_left": "wood", "leg_rear_right": "wood"},
    },
    "car": {
        "plan": CAR_PLAN, "spec": "car",
        "bbox": ([-1.9, -0.1, -1.2], [1.9, 1.75, 1.2]),
        "contacts": [
            ("wheel_front_left", "ground"),
            ("wheel_front_right", "ground"),
            ("wheel_rear_left", "ground"),
            ("wheel_rear_right", "ground"),
            ("cabin", "car_body"),
            ("windshield", "cabin"),
            ("side_window_left", "cabin"),
            ("side_window_right", "cabin"),
        ],
        "materials": {"car_body": "body", "cabin": "accent",
                      "windshield": "glass", "side_window_left": "glass",
                      "side_window_right": "glass",
                      "wheel_front_left": "tire",
                      "wheel_front_right": "tire",
                      "wheel_rear_left": "tire",
                      "wheel_rear_right": "tire",
                      "headlight_left": "accent",
                      "headlight_right": "accent"},
    },
    "cat": {
        "plan": CAT_PLAN, "spec": "cat",
        "bbox": ([-0.93, -0.1, -0.45], [0.91, 2.5, 0.45]),
        "contacts": [(leg, "body") for leg in _legs()] + [
            ("neck", "body"),
            ("head", "neck"),
            ("ear_left", "head"),
            ("ear_right", "head"),
            ("tail", "body"),
        ],
        "materials": {"body": "body", "neck": "body", "head": "body",
                      "ear_left": "body", "ear_right": "body",
                      "leg_front_left": "body", "leg_front_right": "body",
                      "leg_rear_left": "body", "leg_rear_right": "body",
                      "tail": "body"},
    },
    "box_sphere": {
        "plan": BOX_SPHERE_PLAN, "spec": "box_sphere",
        "bbox": ([-0.8, -0.1, -0.8], [0.8, 2.3, 0.8]),
        "contacts": [("box", "ground"), ("sphere", "box")],
        "materials": {"box": "blue", "sphere": "accent"},
    },
}


def get_fixture(name):
    fx = copy.deepcopy(FIXTURES[name])
    return fx
