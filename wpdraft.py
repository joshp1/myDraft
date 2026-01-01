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

 # NEW: selection (0-based, end-exclusive)
    sel_start: int = 0
    sel_end: int = 0

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

    def set_selection(self, a: int, b: int) -> None:
        n = len(self.text)
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        self.sel_start, self.sel_end = a, b

    def toggle_attr_on_selection(self, attr: str) -> None:
        if attr not in ALLOWED_ATTRS:
            raise ValueError(f"Bad attr: {attr}")
        a, b = sorted((self.sel_start, self.sel_end))
        if a == b:
            return

        if self._range_fully_has_attr(a, b, attr):
            self._remove_attr_range(a, b, attr)
        else:
            self._add_attr_range(a, b, attr)

        self._normalize_spans()
        self.dirty = True

# ---------- helpers for toggle ----------

    def _range_fully_has_attr(self, a: int, b: int, attr: str) -> bool:
        """Return True if [a,b) is completely covered by spans containing attr."""
        covered = a
        for sp in sorted(self.spans, key=lambda s: (s.start, s.end)):
            if attr not in sp.attrs:
                continue
            if sp.end <= covered:
                continue
            if sp.start > covered:
                return False
            covered = max(covered, sp.end)
            if covered >= b:
                return True
        return covered >= b

    def _add_attr_range(self, a: int, b: int, attr: str) -> None:
        """Add attr to [a,b), splitting spans as needed to avoid partial overlaps."""
        new_spans: List[Span] = []
        for sp in self.spans:
            # disjoint
            if sp.end <= a or sp.start >= b:
                new_spans.append(sp)
                continue

            # overlap: split into up to 3 pieces
            left_s, left_e = sp.start, max(sp.start, a)
            mid_s, mid_e = max(sp.start, a), min(sp.end, b)
            right_s, right_e = min(sp.end, b), sp.end

            if left_e > left_s:
                new_spans.append(Span(left_s, left_e, set(sp.attrs)))

            if mid_e > mid_s:
                mid_attrs = set(sp.attrs)
                mid_attrs.add(attr)
                new_spans.append(Span(mid_s, mid_e, mid_attrs))

            if right_e > right_s:
                new_spans.append(Span(right_s, right_e, set(sp.attrs)))

        # Also cover gaps where there was no span at all
        # by adding a new span with just {attr} for uncovered parts.
        gaps = self._subtract_covered_by_any_span(a, b)
        for gs, ge in gaps:
            new_spans.append(Span(gs, ge, {attr}))

        self.spans = sorted(new_spans, key=lambda s: (s.start, s.end))

    def _remove_attr_range(self, a: int, b: int, attr: str) -> None:
        """Remove attr from [a,b), splitting spans as needed."""
        new_spans: List[Span] = []
        for sp in self.spans:
            if sp.end <= a or sp.start >= b or attr not in sp.attrs:
                new_spans.append(sp)
                continue

            left_s, left_e = sp.start, max(sp.start, a)
            mid_s, mid_e = max(sp.start, a), min(sp.end, b)
            right_s, right_e = min(sp.end, b), sp.end

            if left_e > left_s:
                new_spans.append(Span(left_s, left_e, set(sp.attrs)))

            if mid_e > mid_s:
                mid_attrs = set(sp.attrs)
                mid_attrs.discard(attr)
                if mid_attrs:
                    new_spans.append(Span(mid_s, mid_e, mid_attrs))
                # else drop it completely

            if right_e > right_s:
                new_spans.append(Span(right_s, right_e, set(sp.attrs)))

        self.spans = sorted(new_spans, key=lambda s: (s.start, s.end))

    def _subtract_covered_by_any_span(self, a: int, b: int) -> List[Tuple[int, int]]:
        """Return gaps in [a,b) that are not covered by ANY existing span (any attrs)."""
        spans = sorted(self.spans, key=lambda s: (s.start, s.end))
        cur = a
        gaps: List[Tuple[int, int]] = []
        for sp in spans:
            if sp.end <= cur:
                continue
            if sp.start >= b:
                break
            if sp.start > cur:
                gaps.append((cur, min(sp.start, b)))
            cur = max(cur, sp.end)
            if cur >= b:
                break
        if cur < b:
            gaps.append((cur, b))
        return [(s, e) for s, e in gaps if e > s]

    def _normalize_spans(self) -> None:
        """Merge adjacent spans with identical attrs. Clamp to text length."""
        self._clamp_spans()
        if not self.spans:
            return
        merged: List[Span] = []
        for sp in sorted(self.spans, key=lambda s: (s.start, s.end)):
            if not sp.attrs:
                continue
            if not merged:
                merged.append(sp)
                continue
            prev = merged[-1]
            if prev.end == sp.start and prev.attrs == sp.attrs:
                merged[-1] = Span(prev.start, sp.end, set(prev.attrs))
            else:
                merged.append(sp)
        self.spans = merged

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
# ---------- curses editor v1 (typing + cursor + save/quit) ----------

import curses

def _build_line_index(text: str):
    """
    Returns (lines, starts)
    lines: list[str] each line WITHOUT trailing '\n' (except empty last)
    starts: list[int] starting absolute index in text for each line
    """
    # Keep '\n' out of lines for simpler rendering
    raw = text.splitlines(True)  # keepends
    lines = []
    starts = []
    pos = 0
    if not raw:
        return [""], [0]
    for chunk in raw:
        starts.append(pos)
        if chunk.endswith("\n"):
            lines.append(chunk[:-1])
            pos += len(chunk)
        else:
            lines.append(chunk)
            pos += len(chunk)
    # If text ends with '\n', splitlines(True) gives last chunk ending in '\n'
    # and no empty line. Add a trailing empty line for cursor placement.
    if text.endswith("\n"):
        starts.append(pos)
        lines.append("")
    return lines, starts

def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def _line_col_from_pos(lines, starts, pos: int):
    # Find last line whose start <= pos (linear is ok for v1)
    i = 0
    for j in range(len(starts)):
        if starts[j] <= pos:
            i = j
        else:
            break
    col = pos - starts[i]
    col = _clamp(col, 0, len(lines[i]))
    return i, col

def _pos_from_line_col(lines, starts, line: int, col: int):
    line = _clamp(line, 0, len(lines) - 1)
    col = _clamp(col, 0, len(lines[line]))
    return starts[line] + col

def _insert(doc: "Document", pos: int, s: str) -> int:
    doc.text = doc.text[:pos] + s + doc.text[pos:]
    doc._clamp_spans()
    doc.dirty = True
    return pos + len(s)

def _delete_backspace(doc: "Document", pos: int) -> int:
    if pos <= 0:
        return 0
    doc.text = doc.text[:pos-1] + doc.text[pos:]
    doc._clamp_spans()
    doc.dirty = True
    return pos - 1

def curses_editor(doc: "Document") -> int:
    """
    Keys:
      Arrows: move
      PgUp/PgDn: scroll page
      Home/End: line start/end
      F7: save
      Ctrl+X or F10: quit
    """
    def _run(stdscr):
        curses.curs_set(1)
        stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

        cursor_pos = 0
        preferred_col = 0
        top_line = 0
        msg = ""

        # Start cursor at end of text (optional)
        cursor_pos = len(doc.text)

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            view_h = max(1, h - 1)  # last line is status bar

            lines, starts = _build_line_index(doc.text)
            cur_line, cur_col = _line_col_from_pos(lines, starts, cursor_pos)

            # Ensure cursor line visible
            if cur_line < top_line:
                top_line = cur_line
            elif cur_line >= top_line + view_h:
                top_line = cur_line - view_h + 1
            top_line = _clamp(top_line, 0, max(0, len(lines) - 1))

            # Render text area
            for row in range(view_h):
                li = top_line + row
                if li >= len(lines):
                    break
                # Clip to width
                s = lines[li]
                if len(s) > w:
                    s = s[:w]
                stdscr.addstr(row, 0, s)

            # Status bar
            name = str(doc.path) if doc.path else "(no file)"
            dirty = "*" if doc.dirty else ""
            status = f"{name}{dirty}  Ln {cur_line+1}, Col {cur_col+1}  F7/Ctrl+S Save  Ctrl+Q Quit"
            if msg:
                # show message at end if space
                status = status[:max(0, w - 1)]
                msg_part = (" | " + msg)[:max(0, w - 1 - len(status))]
                status = status + msg_part
            if len(status) < w:
                status = status + (" " * (w - len(status) - 1))
            stdscr.addstr(h - 1, 0, status[:max(0, w - 1)], curses.A_REVERSE)

            # Place cursor
            screen_y = cur_line - top_line
            screen_x = _clamp(cur_col, 0, max(0, w - 1))
            if 0 <= screen_y < view_h:
                stdscr.move(screen_y, screen_x)

            stdscr.refresh()
            msg = ""

            ch = stdscr.getch()

            # Quit
            if ch in (24,27):  # Ctrl+Q,, ESC
                if doc.dirty:
                    msg = "Unsaved changes. Press Q again to quit."
                    stdscr.refresh()
                    ch2 = stdscr.getch()
                    if ch2 in (curses.KEY_F10, ord('q'), ord('Q'), 17):
                        return 0
                    continue
                return 0

            # Save
            if ch == curses.KEY_F7:  # F7 or Ctrl+S
                if not doc.path:
                    msg = "No path. Use REPL mode to :save <path> once."
                    continue
                try:
                    save_xml(doc.path, doc)
                    msg = "Saved."
                except Exception as e:
                    msg = f"Save error: {e}"
                continue

            # Navigation
            if ch == curses.KEY_LEFT:
                cursor_pos = max(0, cursor_pos - 1)
                cur_line, cur_col = _line_col_from_pos(lines, starts, cursor_pos)
                preferred_col = cur_col
                continue

            if ch == curses.KEY_RIGHT:
                cursor_pos = min(len(doc.text), cursor_pos + 1)
                cur_line, cur_col = _line_col_from_pos(lines, starts, cursor_pos)
                preferred_col = cur_col
                continue

            if ch == curses.KEY_UP:
                new_line = max(0, cur_line - 1)
                cursor_pos = _pos_from_line_col(lines, starts, new_line, preferred_col)
                continue

            if ch == curses.KEY_DOWN:
                new_line = min(len(lines) - 1, cur_line + 1)
                cursor_pos = _pos_from_line_col(lines, starts, new_line, preferred_col)
                continue

            if ch == curses.KEY_HOME:
                cursor_pos = starts[cur_line]
                preferred_col = 0
                continue

            if ch == curses.KEY_END:
                cursor_pos = starts[cur_line] + len(lines[cur_line])
                preferred_col = len(lines[cur_line])
                continue

            if ch == curses.KEY_NPAGE:  # Page Down
                top_line = min(max(0, len(lines) - 1), top_line + view_h)
                cur_line = min(len(lines) - 1, top_line)
                cursor_pos = _pos_from_line_col(lines, starts, cur_line, preferred_col)
                continue

            if ch == curses.KEY_PPAGE:  # Page Up
                top_line = max(0, top_line - view_h)
                cur_line = top_line
                cursor_pos = _pos_from_line_col(lines, starts, cur_line, preferred_col)
                continue

            # Editing
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                cursor_pos = _delete_backspace(doc, cursor_pos)
                cur_line, cur_col = _line_col_from_pos(lines, starts, cursor_pos)
                preferred_col = cur_col
                continue

            if ch in (10, 13):  # Enter
                cursor_pos = _insert(doc, cursor_pos, "\n")
                preferred_col = 0
                continue

            if ch == 9:  # Tab -> 4 spaces
                cursor_pos = _insert(doc, cursor_pos, "    ")
                cur_line, cur_col = _line_col_from_pos(lines, starts, cursor_pos)
                preferred_col = cur_col
                continue

            # Printable characters
            if 32 <= ch <= 126:
                cursor_pos = _insert(doc, cursor_pos, chr(ch))
                cur_line, cur_col = _line_col_from_pos(lines, starts, cursor_pos)
                preferred_col = cur_col
                continue

            # Ignore everything else
            msg = f"Key {ch} ignored."

    return curses.wrapper(_run)


def main(argv: List[str]) -> int:
    use_curses = "--curses" in argv

    # first non-flag argument is the file path
    file_arg = next((a for a in argv[1:] if not a.startswith("-")), None)

    doc = Document()
    if file_arg:
        p = Path(file_arg).expanduser()
        if p.exists():
            doc = load_xml(p)
        else:
            doc.path = p

    if use_curses:
        return curses_editor(doc)

    # fallback to existing REPL
    return repl(file_arg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
