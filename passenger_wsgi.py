import os
from pathlib import Path
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
application = app  # Passenger looks for "application"

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

LATEST_WPX = DATA / "latest.wpx"
LATEST_MD = DATA / "latest.md"
LATEST_BBC = DATA / "latest.bbcode"

def write_text_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)

def save_wpx(text: str, spans: list[dict]) -> str:
    # Very small WPX writer. attrs stored in "a", start in "s", end in "e".
    import xml.etree.ElementTree as ET
    root = ET.Element("wpx", {"version": "1"})
    t = ET.SubElement(root, "text")
    t.text = text
    se = ET.SubElement(root, "spans")
    for sp in spans:
        s = int(sp["s"]); e = int(sp["e"])
        a = " ".join(sorted(set(sp.get("a", []))))
        ET.SubElement(se, "span", {"s": str(s), "e": str(e), "a": a})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

def export_markdown(text: str, spans: list[dict]) -> str:
    # Simple event-based export for attrs b/i/u. Assumes no partial overlap.
    n = len(text)
    opens = [[] for _ in range(n + 1)]
    closes = [[] for _ in range(n + 1)]

    def ot(a): return {"b": "**", "i": "*", "u": "<u>"}[a]
    def ct(a): return {"b": "**", "i": "*", "u": "</u>"}[a]

    for sp in spans:
        s = int(sp["s"]); e = int(sp["e"])
        for a in sorted(set(sp.get("a", []))):
            opens[s].append((e - s, a))
            closes[e].append((e - s, a))

    for i in range(n + 1):
        opens[i].sort(key=lambda t: (-t[0], t[1]))
        closes[i].sort(key=lambda t: (t[0], t[1]))

    out = []
    for i, ch in enumerate(text):
        for _, a in opens[i]:
            out.append(ot(a))
        out.append(ch)
        for _, a in closes[i + 1]:
            out.append(ct(a))
    return "".join(out)

def export_bbcode(text: str, spans: list[dict]) -> str:
    n = len(text)
    opens = [[] for _ in range(n + 1)]
    closes = [[] for _ in range(n + 1)]

    def ot(a): return {"b": "[b]", "i": "[i]", "u": "[u]"}[a]
    def ct(a): return {"b": "[/b]", "i": "[/i]", "u": "[/u]"}[a]

    for sp in spans:
        s = int(sp["s"]); e = int(sp["e"])
        for a in sorted(set(sp.get("a", []))):
            opens[s].append((e - s, a))
            closes[e].append((e - s, a))

    for i in range(n + 1):
        opens[i].sort(key=lambda t: (-t[0], t[1]))
        closes[i].sort(key=lambda t: (t[0], t[1]))

    out = []
    for i, ch in enumerate(text):
        for _, a in opens[i]:
            out.append(ot(a))
        out.append(ch)
        for _, a in closes[i + 1]:
            out.append(ct(a))
    return "".join(out)

@app.get("/")
def home():
    return send_file(BASE / "index.html")

@app.get("/load")
def load_latest():
    if not LATEST_WPX.exists():
        return jsonify({"text": "", "spans": []})
    # Minimal load: parse WPX back to JSON
    import xml.etree.ElementTree as ET
    root = ET.parse(str(LATEST_WPX)).getroot()
    text = (root.findtext("text") or "")
    spans = []
    se = root.find("spans")
    if se is not None:
        for el in se.findall("span"):
            s = el.get("s"); e = el.get("e")
            a = (el.get("a") or "").split()
            if s is None or e is None:
                continue
            spans.append({"s": int(s), "e": int(e), "a": a})
    return jsonify({"text": text, "spans": spans})

@app.post("/save")
def save():
    data = request.get_json(force=True)
    text = data.get("text", "")
    spans = data.get("spans", [])
    wpx = save_wpx(text, spans)

    write_text_atomic(LATEST_WPX, wpx)
    write_text_atomic(LATEST_MD, export_markdown(text, spans))
    write_text_atomic(LATEST_BBC, export_bbcode(text, spans))

    return jsonify({"ok": True})

