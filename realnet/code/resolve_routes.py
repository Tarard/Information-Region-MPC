"""Write a copy of a SUMO route file where every vehicle has an inline <route edges=...> child."""
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
root = ET.parse(src).getroot()
named = {r.get("id"): r.get("edges") for r in root.findall("route") if r.get("id")}
kinds = Counter()
for v in root.findall("vehicle"):
    ref = v.attrib.pop("route", None)
    child = v.find("route")
    if child is not None and child.get("edges"):
        kinds["inline"] += 1
    elif ref in named:
        ET.SubElement(v, "route", {"edges": named[ref]})
        kinds["named"] += 1
    else:
        kinds["unresolved"] += 1
dst.parent.mkdir(parents=True, exist_ok=True)
ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
print(dict(kinds), "named routes:", len(named), "vehicles:", len(root.findall("vehicle")))
