# tests/image3d/image3d_fixtures.py
# Deterministic reference images for the image -> 3D regression suite.
# Drawn procedurally (PIL, fixed seeds) so every case is reproducible
# offline: several categories, several views, flat / gradient / cluttered
# backgrounds, display platforms, occlusion, and subjects that have NO
# dedicated builder (the system must still reconstruct them).
#
# Real photos live next to these: the user's Urus screenshot
# (real_urus.png) and the synthetic side-view car (car_side_platform.png).
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
W, H = 480, 320


def _canvas(bg=(235, 238, 240), gradient=None):
    img = Image.new("RGB", (W, H), bg)
    if gradient:
        top, bot = gradient
        d = ImageDraw.Draw(img)
        for y in range(H):
            t = y / (H - 1)
            d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * t)
                                                for a, b in zip(top, bot)))
    return img, ImageDraw.Draw(img)


def car_side(d, x0=70, y_ground=250, L=340, body=(200, 30, 30), facing=1,
             suv=False):
    h = 70 if not suv else 95
    belt = y_ground - 35 - h * 0.55
    d.rounded_rectangle([x0, belt, x0 + L, y_ground - 28], 14, fill=body)
    cab0 = x0 + (L * 0.28 if facing > 0 else L * 0.22)
    cab1 = x0 + (L * 0.70 if facing > 0 else L * 0.78)
    roof = belt - h * 0.75
    d.polygon([(cab0, belt), (cab0 + 30 * facing if facing > 0 else cab0 + 45, roof),
               (cab1 - 45 if facing > 0 else cab1 - 30, roof), (cab1, belt)],
              fill=body)
    d.polygon([(cab0 + 12, belt - 4), (cab0 + 38, roof + 8),
               (cab1 - 50, roof + 8), (cab1 - 14, belt - 4)],
              fill=(150, 200, 225))
    r = 30
    for wx in (x0 + L * 0.2, x0 + L * 0.8):
        d.ellipse([wx - r, y_ground - 2 * r, wx + r, y_ground], fill=(25, 25, 28))
        d.ellipse([wx - 12, y_ground - r - 12, wx + 12, y_ground - r + 12],
                  fill=(150, 150, 155))
    lx = x0 + L - 10 if facing > 0 else x0 + 2
    d.rectangle([lx, belt + 10, lx + 8, belt + 22], fill=(255, 245, 200))


def make_all():
    os.makedirs(INPUTS, exist_ok=True)
    rng = random.Random(7)
    out = {}

    img, d = _canvas((0, 200, 200))
    d.ellipse([60, 225, 420, 300], fill=(200, 30, 30))
    car_side(d, body=(30, 28, 34))
    out["car_side_platform"] = img

    img, d = _canvas(gradient=((180, 200, 230), (90, 100, 120)))
    for _ in range(9):  # clutter: boxes and blobs around the subject
        x, y = rng.randint(0, W - 40), rng.randint(0, 80)
        d.rectangle([x, y, x + rng.randint(15, 50), y + rng.randint(10, 30)],
                    fill=tuple(rng.randint(60, 250) for _ in range(3)))
    car_side(d, body=(40, 90, 200), facing=-1, suv=True)
    out["car_suv_clutter"] = img

    img, d = _canvas((245, 245, 245))
    car_side(d, body=(230, 200, 40))
    d.rectangle([60, 150, 170, 270], fill=(120, 80, 50))  # occluder (crate)
    out["car_occluded"] = img

    img, d = _canvas((250, 250, 250))
    d.rectangle([90, 120, 390, 140], fill=(140, 90, 50))
    for x in (105, 360):
        d.rectangle([x, 140, x + 16, 270], fill=(120, 75, 40))
    for x in (140, 325):
        d.rectangle([x, 140, x + 12, 245], fill=(100, 62, 35))
    out["table_angle"] = img

    img, d = _canvas((240, 240, 235))
    d.rectangle([180, 170, 310, 186], fill=(90, 60, 40))   # seat
    d.rectangle([180, 60, 196, 186], fill=(90, 60, 40))    # back post
    d.rectangle([180, 60, 200, 120], fill=(90, 60, 40))
    for x in (182, 294):
        d.rectangle([x, 186, x + 14, 285], fill=(80, 52, 34))
    out["chair_side"] = img

    img, d = _canvas(gradient=((150, 200, 240), (210, 230, 245)))
    d.rectangle([0, 270, W, H], fill=(80, 150, 60))         # grass floor
    d.rectangle([130, 150, 350, 270], fill=(230, 220, 190))
    d.polygon([(110, 152), (370, 152), (240, 70)], fill=(160, 50, 40))
    d.rectangle([220, 205, 260, 270], fill=(90, 60, 40))
    d.rectangle([150, 175, 195, 210], fill=(140, 190, 230))
    d.rectangle([285, 175, 330, 210], fill=(140, 190, 230))
    out["house_front"] = img

    img, d = _canvas((225, 230, 238))
    d.ellipse([140, 150, 330, 230], fill=(230, 140, 50))     # torso
    d.ellipse([300, 110, 370, 175], fill=(230, 140, 50))     # head
    d.polygon([(308, 122), (318, 88), (332, 116)], fill=(230, 140, 50))
    d.polygon([(338, 116), (352, 88), (362, 124)], fill=(230, 140, 50))
    for x in (160, 190, 270, 300):
        d.rectangle([x, 210, x + 14, 280], fill=(220, 130, 45))
    d.line([(145, 180), (100, 130), (90, 100)], fill=(220, 130, 45), width=12)
    out["cat_side"] = img

    img, d = _canvas((60, 64, 70))
    prof = []
    for i in range(41):
        y = 60 + i * 5
        t = i / 40.0
        r = 28 + 55 * math.sin(math.pi * min(1, t * 1.15)) * (0.6 + 0.4 * t)
        if t < 0.15:
            r = 30 + 10 * t
        prof.append((r, y))
    pts = [(240 - r, y) for r, y in prof] + [(240 + r, y) for r, y in prof[::-1]]
    d.polygon(pts, fill=(60, 120, 170))
    out["vase_front"] = img

    def teapot(d, col):
        d.ellipse([170, 130, 320, 250], fill=col)
        d.rectangle([215, 112, 275, 132], fill=col)
        d.ellipse([233, 98, 257, 116], fill=col)
        d.polygon([(310, 190), (380, 140), (390, 150), (318, 215)], fill=col)
        d.arc([125, 150, 195, 225], 90, 270, fill=col, width=14)
        d.ellipse([200, 245, 290, 258], fill=col)

    img, d = _canvas((238, 236, 230))   # teapot: no dedicated builder
    teapot(d, (70, 110, 150))
    out["teapot_side"] = img

    img, d = _canvas((238, 236, 230))   # same teapot, white on cream:
    teapot(d, (240, 240, 245))          # too little contrast to segment
    out["lowcontrast_teapot"] = img

    img, d = _canvas((250, 246, 240))   # guitar: no dedicated builder
    d.ellipse([90, 140, 230, 270], fill=(170, 90, 40))
    d.ellipse([190, 160, 290, 245], fill=(170, 90, 40))
    d.rectangle([280, 190, 430, 212], fill=(60, 40, 25))
    d.rectangle([425, 182, 462, 220], fill=(60, 40, 25))
    d.ellipse([150, 185, 190, 225], fill=(40, 25, 15))
    out["guitar_side"] = img

    paths = {}
    for name, im in out.items():
        p = os.path.join(INPUTS, f"{name}.png")
        im.filter(ImageFilter.SMOOTH).save(p)
        paths[name] = p
    return paths


if __name__ == "__main__":
    for k, v in make_all().items():
        print(k, v)
