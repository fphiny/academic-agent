import re
from typing import List

def _normalize_ws(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def _normalize_block(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = text.split("\n")

    out: List[str] = []
    prev_blank = False
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if out and not prev_blank:
                out.append("")
            prev_blank = True
            continue
        out.append(line)
        prev_blank = False

    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()

    return "\n".join(out)


def _normalize_code(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)