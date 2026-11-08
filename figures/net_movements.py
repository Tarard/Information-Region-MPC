"""Read signal movements from the SUMO network for the phase diagram."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "realnet"
NET = ROOT / "v3" / "inputs" / "lyg_safe.net.xml"
SIGNALS = ROOT / "inputs" / "mix_signals.json"


def _bearing(p, q):
    """Compass bearing in degrees (0 = north, 90 = east) from p to q."""
    return math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) % 360


def _shape(s):
    return [tuple(map(float, xy.split(","))) for xy in s.split()]


def load():
    """Return {tls: {"xy": (x, y), "links": {index: movement}, "stages": [...]}}."""
    root = ET.parse(NET).getroot()
    edges = {}
    for e in root.iter("edge"):
        if e.get("function") == "internal":
            continue
        shp = e.get("shape") or e.find("lane").get("shape")
        edges[e.get("id")] = _shape(shp)
    junctions = {j.get("id"): (float(j.get("x")), float(j.get("y"))) for j in root.iter("junction")}
    sig = json.loads(SIGNALS.read_text(encoding="utf-8"))["signals"]
    out = {tls: {"xy": junctions[tls], "links": {}, "stages": cfg["stages"]} for tls, cfg in sig.items()}
    for c in root.iter("connection"):
        tls = c.get("tl")
        if tls not in out or c.get("linkIndex") is None:
            continue
        shp = edges[c.get("from")]
        # Approach direction: where the vehicles come from, seen from the stop line.
        head = _bearing(shp[-2], shp[-1])
        out[tls]["links"][int(c.get("linkIndex"))] = {
            "from": c.get("from"), "to": c.get("to"), "dir": c.get("dir"), "heading": head,
        }
    return out


if __name__ == "__main__":
    data = load()
    for tls, d in data.items():
        print(tls, len(d["stages"]), [round(v) for v in d["xy"]], len(d["links"]))
