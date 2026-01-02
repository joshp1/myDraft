import sys
import xml.etree.ElementTree as ET

p = sys.argv[1]
tree = ET.parse(p)
root = tree.getroot()

bad = []
for sp in root.findall(".//span"):
    if "s" not in sp.attrib or "e" not in sp.attrib:
        bad.append((getattr(sp, "sourceline", None), dict(sp.attrib)))

print("bad spans:", len(bad))
for line, attrib in bad[:50]:
    print("line", line, attrib)
