import re
s = open("tests/visual_geometry/fixtures.py", encoding="utf-8").read()
i = s.find("CAT_PLAN")
print(s[i:i + 400])
print("======")
for m in re.finditer("cat", s):
    pass
for m in re.finditer('"cat":', s):
    print("CAT ENTRY at line", s[:m.start()].count("\n") + 1)
    print(s[m.start():m.start() + 1200])
