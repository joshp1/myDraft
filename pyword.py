#!/usr/bin/env python3
"""
wpdraft.py
Minimal drafting core:
- In-memory document (text + non-overlapping spans)
- Save/load XML
- Export Markdown
- Command-loop UI (no curses)

Design limits (intentional for v1):
- Spans may be nested but must NOT partially overlap.
- Underline exports as HTML <u>...</u> inside Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple, Optional
import xml.etree.ElementTree as ET
import html
import sys
import os
import tempfile


# ---------- Document model ----------

ALLOWED_ATTRS = {"b", "u", "s"}  # bold, underline, strike


@dataclass(order=True)
class Span:
    start: int
    end: int
    attrs: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0 or self.end < self.start:
            raise ValueError(f"Invalid span range: {self.start}-{self.end}")
        bad = set(self.attrs) - ALLOWED_ATTRS
        if bad:
            raise ValueError(f"Unknown attrs: {bad}")


@dataclass
class Document:
    text: str = ""
    spans: List[Span] = field(default_factory=list)
    dirty: bool = False
    path: Optional[Path] = None

    def set_text(self, new_text: str) -> None:
        self.text = new_text
        self._clamp_spans()
        self.dirty = True

    def append_text(self, more: str) -> None:
        self.text += more
        self.dirty = True

    def add_span(self, start: int, end: int, attrs: Set[str]) -> None:
        if start == end:
            return
        if start < 0 or end > len(self.text) or start > end:
            raise ValueError("Span out of bounds.")
        span = Span(start, end, set(attrs))
        self._validate_no_partial_overlap(span)
        self.spans.append(span)
        self.spans.sort(key=lambda s: (s.start, s.end))
        self.dirty = True

    def clear_spans(self) -> None:
        self.spans.clear()
        self.dirty = True

    def _clamp_spans(self) -> None:
        n = len(self.text)
        clamped: List[Span] = []
        for sp in self.spans:
            s = max(0, min(sp.start, n))
            e = max(0, min(sp.end, n))
            if e > s and sp.attrs:
                clamped.append(Span(s, e, set(sp.attrs)))
        self.spans = sorted(clamped, key=lambda s: (s.start, s.end))

    def _validate_no_partial_overlap(self, new_span: Span) -> None:
        # Allow disjoint or nested, forbid partial overlap (A starts inside B but ends outside).
        for sp in self.spans:
            if new_span.end <= sp.start or new_span.start >= sp.end:
                continue  # disjoint
            # overlap exists
            nested1 = new_span.start >= sp.start and new_span.end <= sp.end
            nested2 = sp.start >= new_span.start and sp.end <= new_span.end
            if nested1 or nested2:
                continue
            raise ValueError(
                f"Partial overlap not allowed: new {new_span.start}-{new_span.end} "
                f"existing {sp.start}-{sp.end}"
            )


# ---------- XML format ----------

# Format:
# <wpx version="1">
#   <text>...</text>
#   <spans>
#     <span s="10" e="25" a="b u"/>
#   </spans>
# </wpx>

def load_xml(path: Path) -> Document:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "wpx":
        raise ValueError("Not a wpx document.")
    version = root.attrib.get("version", "1")
    if version != "1":
        raise ValueError(f"Unsupported version: {version}")

    text_el = root.find("text")
    text = text_el.text if (text_el is not None and text_el.text is not None) else ""

    doc = Document(text=text, spans=[], dirty=False, path=path)

    spans_el = root.find("spans")
    if spans_el is not None:
        for sp_el in spans_el.findall("span"):
            s = int(sp_el.attrib["s"])
            e = int(sp_el.attrib["e"])
            a = sp_el.attrib.get("a", "").strip()
            attrs = set(a.split()) if a else set()
            if attrs:
                doc.add_span(s, e, attrs)
                doc.dirty = False  # add_span marks dirty; reset after load
    doc._clamp_spans()
    doc.dirty = False
    return doc


def save_xml(path: Path, doc: Document) -> None:
    root = ET.Element("wpx", {"version": "1"})
    text_el = ET.SubElement(root, "text")
    text_el.text = doc.text

    spans_el = ET.SubElement(root, "spans")
    for sp in sorted(doc.spans, key=lambda s: (s.start, s.end)):
        a = " ".join(sorted(sp.attrs))
        ET.SubElement(spans_el, "span", {"s": str(sp.start), "e": str(sp.end), "a": a})

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    # Atomic write: write to temp then rename.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(xml_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass

    doc.dirty = False
    doc.path = path


# ---------- Markdown export ----------

def export_markdown(doc: Document) -> str:
    """
    Export spans into Markdown.
    Rules:
    - Bold -> **...**
    - Strike -> ~~...~~
    - Underline -> <u>...</u>  (Markdown has no standard underline)
    Overlap rule: no partial overlap (enforced).
    """
    text = doc.text
    n = len(text)

    # Create open/close events by position.
    opens: List[List[Tuple[int, str]]] = [[] for _ in range(n + 1)]
    closes: List[List[Tuple[int, str]]] = [[] for _ in range(n + 1)]

    def open_token(attr: str) -> str:
        if attr == "b":
            return "**"
        if attr == "s":
            return "~~"
        if attr == "u":
            return "<u>"
        raise ValueError(attr)

    def close_token(attr: str) -> str:
        if attr == "b":
            return "**"
        if attr == "s":
            return "~~"
        if attr == "u":
            return "</u>"
        raise ValueError(attr)

    # For stable nesting: close inner first, open outer first.
    # We'll sort opens by longer spans first (outer first), closes by shorter spans first (inner first).
    for sp in doc.spans:
        for attr in sorted(sp.attrs):
            opens[sp.start].append((sp.end - sp.start, attr))
            closes[sp.end].append((sp.end - sp.start, attr))

    for i in range(n + 1):
        opens[i].sort(key=lambda t: (-t[0], t[1]))   # outer first
        closes[i].sort(key=lambda t: (t[0], t[1]))   # inner first

    out: List[str] = []
    for i, ch in enumerate(text):
        # emit opens at i
        for _, attr in opens[i]:
            out.append(open_token(attr))
        out.append(ch)
        # emit closes after ch at i+1 boundary? We stored closes at end index; handle when i+1 reached.
        for _, attr in closes[i + 1]:
            out.append(close_token(attr))

    # closes at position 0 (if any) handled by i=-1 not possible, but ends at 0 makes no sense.
    # also handle closes at n if text empty:
    if n == 0:
        for _, attr in opens[0]:
            out.append(open_token(attr))
        for _, attr in closes[0]:
            out.append(close_token(attr))

    return "".join(out)


# ---------- Minimal command UI ----------

HELP = """Commands:
  :help                         Show help
  :show                         Show current text (with indexes)
  :set                           Replace text (multiline; end with a single dot '.' line)
  :append                        Append text (multiline; end with '.' line)
  :bold <start> <end>            Apply bold span
  :under <start> <end>           Apply underline span (exports as <u>..</u>)
  :strike <start> <end>          Apply strike span
  :clearspans                    Remove all spans
  :save [path]                   Save XML (default: current path)
  :open <path>                   Load XML
  :export [path]                 Export Markdown (default: same name .md)
  :status                        Show doc status
  :quit                          Exit (warns if dirty)

Indexing: 0-based, end is exclusive. Example: :bold 10 25
"""

def show_with_indexes(doc: Document) -> None:
    s = doc.text
    print(f"Length: {len(s)}")
    if not s:
        print("(empty)")
        return
    # Print text lines with line numbers and a simple char index ruler every 10 chars.
    # Keep it simple and safe for terminals.
    print(s)

def read_multiline(prompt: str) -> str:
    print(prompt)
    lines: List[str] = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines)

def repl(initial_path: Optional[str]) -> int:
    doc = Document()
    if initial_path:
        p = Path(initial_path).expanduser()
        if p.exists():
            doc = load_xml(p)
        else:
            doc.path = p

    print("wpdraft core (no curses). Type :help")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cmd = ":quit"

        if not cmd:
            continue

        if cmd == ":help":
            print(HELP)
            continue

        if cmd == ":show":
            show_with_indexes(doc)
            print("\nSpans:")
            for sp in doc.spans:
                print(f"  {sp.start}-{sp.end} {sorted(sp.attrs)}")
            continue

        if cmd == ":set":
            txt = read_multiline("Enter text. End with a single '.' line.")
            doc.set_text(txt)
            continue

        if cmd == ":append":
            txt = read_multiline("Append text. End with a single '.' line.")
            if doc.text and txt:
                doc.append_text("\n" + txt)
            else:
                doc.append_text(txt)
            doc.dirty = True
            continue

        if cmd.startswith(":bold ") or cmd.startswith(":under ") or cmd.startswith(":strike "):
            parts = cmd.split()
            if len(parts) != 3:
                print("Usage: :bold <start> <end>")
                continue
            _, a, b = parts
            try:
                start = int(a); end = int(b)
                if cmd.startswith(":bold "):
                    doc.add_span(start, end, {"b"})
                elif cmd.startswith(":under "):
                    doc.add_span(start, end, {"u"})
                else:
                    doc.add_span(start, end, {"s"})
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd == ":clearspans":
            doc.clear_spans()
            continue

        if cmd.startswith(":open "):
            _, path = cmd.split(maxsplit=1)
            try:
                doc = load_xml(Path(path).expanduser())
                print(f"Loaded {doc.path}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith(":save"):
            parts = cmd.split(maxsplit=1)
            path = None
            if len(parts) == 2:
                path = parts[1]
            p = Path(path).expanduser() if path else doc.path
            if not p:
                print("No path. Use :save <path>")
                continue
            try:
                save_xml(p, doc)
                print(f"Saved {p}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith(":export"):
            parts = cmd.split(maxsplit=1)
            out_path = None
            if len(parts) == 2:
                out_path = Path(parts[1]).expanduser()
            else:
                if doc.path:
                    out_path = doc.path.with_suffix(".md")
            if not out_path:
                print("No export path. Use :export <path>")
                continue
            try:
                md = export_markdown(doc)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(md, encoding="utf-8")
                print(f"Exported {out_path}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd == ":status":
            print(f"path: {doc.path}")
            print(f"dirty: {doc.dirty}")
            print(f"length: {len(doc.text)}")
            print(f"spans: {len(doc.spans)}")
            continue

        if cmd == ":quit":
            if doc.dirty:
                ans = input("Unsaved changes. Quit anyway? (y/N) ").strip().lower()
                if ans != "y":
                    continue
            return 0

        print("Unknown command. Type :help")

def main(argv: List[str]) -> int:
    initial_path = argv[1] if len(argv) > 1 else None
    return repl(initial_path)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
