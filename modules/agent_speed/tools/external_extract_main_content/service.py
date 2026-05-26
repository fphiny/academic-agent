from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from .html_table import process_html_table_be, json_to_multiline_text
except Exception:
    from html_table import process_html_table_be, json_to_multiline_text


TARGET_CHUNK_CHARS = 1500
MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 40
MIN_MEANINGFUL = 10
MIN_CANDIDATE_TEXT = 80

_NONRENDER_TAGS = frozenset(
    [
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "object",
        "embed",
        "applet",
        "input",
        "button",
        "select",
        "option",
        "textarea",
    ]
)

_BLOCK_TAGS = frozenset(
    [
        "div",
        "section",
        "article",
        "main",
        "aside",
        "header",
        "footer",
        "nav",
        "p",
        "ul",
        "ol",
        "li",
        "dl",
        "table",
        "pre",
        "code",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    ]
)

_TABLE_PLACEHOLDER_ATTR = "data-preprocessed-table"


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


def _strip_markdown_headings(text: str) -> str:
    text = _normalize_block(text)
    if not text:
        return ""

    kept: List[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*#{1,6}\s+.+$", line):
            continue
        kept.append(line)

    return _normalize_block("\n".join(kept))


def _is_meaningful(text: str) -> bool:
    t = _normalize_block(text)
    if not t or len(t) < MIN_MEANINGFUL:
        return False
    tokens = t.split()
    if len(tokens) >= 3 and len(set(tokens)) == 1:
        return False
    return True


def _make_chunk_id(url: str, idx: int, text: str) -> str:
    h = hashlib.md5(f"{url}|{idx}|{text[:64]}".encode()).hexdigest()[:8]
    return f"chunk_{idx:04d}_{h}"


def _new_tag_like(node: Tag, name: str) -> Tag:
    cur: Any = node
    while getattr(cur, "parent", None) is not None:
        cur = cur.parent
    if isinstance(cur, BeautifulSoup):
        return cur.new_tag(name)
    return BeautifulSoup("", "html.parser").new_tag(name)


def _is_preprocessed_table(node: Any) -> bool:
    return isinstance(node, Tag) and node.get(_TABLE_PLACEHOLDER_ATTR) == "1"


def _contains_table_like(node: Any) -> bool:
    if not isinstance(node, Tag):
        return False
    if _is_preprocessed_table(node):
        return True
    if node.find("table"):
        return True
    if node.find(attrs={_TABLE_PLACEHOLDER_ATTR: "1"}):
        return True
    return False


def _text_len(node: Tag) -> int:
    try:
        return len(_normalize_ws(node.get_text(" ", strip=True)))
    except Exception:
        return 0


def _link_text_len(node: Tag) -> int:
    try:
        return sum(len(_normalize_ws(a.get_text(" ", strip=True))) for a in node.find_all("a"))
    except Exception:
        return 0


def _extract_block_text(node: Tag) -> str:
    try:
        if _contains_table_like(node):
            return ""
        direct_blocks = [
            c for c in node.children if isinstance(c, Tag) and (c.name or "").lower() in _BLOCK_TAGS
        ]
        sep = "\n" if node.find("br") or len(direct_blocks) >= 2 else " "
        return _normalize_block(node.get_text(sep, strip=True))
    except Exception:
        return ""


def _split_sentences(text: str) -> List[str]:
    text = _normalize_block(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
    parts = [_normalize_block(p) for p in parts if _normalize_block(p)]
    return parts if parts else [text]


def _hard_slice(text: str, limit: int) -> List[str]:
    text = _normalize_block(text)
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces: List[str] = []
    cur = 0
    while cur < len(text):
        end = min(cur + limit, len(text))
        if end < len(text):
            cut = text.rfind(" ", cur, end)
            if cut <= cur + max(50, limit // 3):
                cut = end
        else:
            cut = end

        piece = text[cur:cut].strip()
        if piece:
            pieces.append(piece)
        cur = cut
    return pieces


def _split_paragraphs(text: str) -> List[str]:
    text = _normalize_block(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _find_table_caption(table: Tag, table_idx: int = 0) -> str:
    caption = ""
    cap = table.find("caption")
    if cap:
        caption = _normalize_ws(cap.get_text(" ", strip=True))
    if caption:
        return caption

    prev = table.find_previous_sibling()
    if prev and isinstance(prev, Tag) and (prev.name or "").lower() == "p":
        prev_text = _normalize_ws(prev.get_text(" ", strip=True))
        if prev_text and len(prev_text) <= 80:
            return prev_text

    return f"table {table_idx}" if table_idx > 0 else ""


def _leaf_score(text_len: int, link_len: int, full_text: str, depth: int, tag_name: str) -> float:
    if text_len <= 0:
        return 0.0

    link_ratio = link_len / max(text_len, 1)
    if link_ratio >= 0.7:
        return 0.0

    nav_chars = sum(1 for c in full_text if c in ">|/\\")
    param_hits = len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_,]+", full_text))
    nav_density = (nav_chars + param_hits * 5) / max(text_len, 1)
    if nav_density >= 0.08:
        return 0.0

    tokens = full_text.split()
    n_tok = max(len(tokens), 1)
    uniq = len(set(tokens))
    punct = sum(1 for c in full_text if c in ",.!?;:。！？")

    volume = min(text_len / 250.0, 1.0)
    link_sc = 1.0 - min(link_ratio / 0.7, 1.0)
    uniq_sc = min((uniq / n_tok) * 1.2, 1.0)
    sent_sc = min((punct / max(text_len, 1)) * 14.0, 1.0)
    depth_sc = min(depth / 10.0, 1.0)

    score = (
        volume * 0.30
        + link_sc * 0.28
        + uniq_sc * 0.18
        + sent_sc * 0.14
        + depth_sc * 0.10
    )

    if tag_name in ("header", "footer", "nav", "aside"):
        score *= 0.55

    has_end = any(c in full_text for c in ".!?。！？")
    short_ratio = sum(1 for t in tokens if len(t) <= 4) / n_tok
    if not has_end and short_ratio >= 0.65 and text_len < 150:
        score *= 0.65

    return max(0.0, min(score, 1.0))


def _node_score(node: Tag) -> float:
    tag_name = (node.name or "").lower()
    depth = sum(1 for _ in node.parents if isinstance(_, Tag))
    html_len = len(str(node))
    text_len = _text_len(node)
    link_len = _link_text_len(node)
    text = _normalize_ws(node.get_text(" ", strip=True))

    if text_len == 0:
        return 0.0

    base = _leaf_score(text_len, link_len, text, depth, tag_name)

    block_children = [
        c for c in node.children if isinstance(c, Tag) and (c.name or "").lower() in _BLOCK_TAGS
    ]
    if block_children:
        total_w = 0.0
        weighted = 0.0
        for child in block_children:
            ct = _text_len(child)
            if ct <= 0:
                continue
            sc = _leaf_score(
                ct,
                _link_text_len(child),
                _normalize_ws(child.get_text(" ", strip=True)),
                depth + 1,
                (child.name or "").lower(),
            )
            weighted += sc * ct
            total_w += ct
        if total_w > 0:
            base = (base * 0.35) + ((weighted / total_w) * 0.65)

    heading_count = len(node.find_all(["h1", "h2", "h3"]))
    para_count = len(node.find_all("p"))
    if heading_count > 0:
        base = min(base * 1.10, 1.0)
    if para_count >= 2:
        base = min(base * 1.08, 1.0)

    text_density = text_len / max(html_len, 1)
    if text_density < 0.035:
        base *= 0.75

    return max(0.0, min(base, 1.0))


def _looks_like_html_markup(text: str) -> bool:
    if not text:
        return False
    sample = text[:4000]
    if re.search(r"<\s*(html|body|div|section|article|main|p|h1|h2|h3|table|ul|li)\b", sample, re.I):
        return True
    if re.search(r"</\s*(html|body|div|section|article|main|p|h1|h2|h3|table|ul|li)\s*>", sample, re.I):
        return True
    return False


class NormalizeLayer:
    PARSER_CHAIN = ["lxml", "html5lib", "html.parser"]

    @classmethod
    def parse(cls, raw_html: str | bytes, url: str = "") -> Tuple[BeautifulSoup, str]:
        html_str, enc = cls._decode(raw_html)
        return cls._parse_with_fallback(html_str), enc

    @classmethod
    def _decode(cls, raw: str | bytes) -> Tuple[str, str]:
        if isinstance(raw, str):
            return raw, "utf-8"

        snippet = raw[:4096].decode("ascii", errors="ignore")
        m = re.search(r'charset=["\']?([\w\-]+)', snippet, re.IGNORECASE)
        if m:
            try:
                return raw.decode(m.group(1), errors="replace"), m.group(1)
            except Exception:
                pass

        try:
            import chardet

            det = chardet.detect(raw[:8192])
            enc = det.get("encoding") or "utf-8"
            if (det.get("confidence") or 0) > 0.7:
                return raw.decode(enc, errors="replace"), enc
        except Exception:
            pass

        return raw.decode("utf-8", errors="replace"), "utf-8"

    @classmethod
    def _parse_with_fallback(cls, html: str) -> BeautifulSoup:
        for parser in cls.PARSER_CHAIN:
            try:
                return BeautifulSoup(html, parser)
            except Exception:
                continue
        return BeautifulSoup(html, "html.parser")

    @classmethod
    def extract_meta(cls, soup: BeautifulSoup) -> Dict[str, str]:
        meta: Dict[str, str] = {}

        if soup.title and soup.title.string:
            meta["title"] = _normalize_ws(soup.title.string)

        for tag in soup.find_all("meta"):
            prop = (tag.get("property", "") or tag.get("name", "")).lower()
            content = tag.get("content", "")
            if not prop or not content:
                continue
            if prop in ("og:title", "twitter:title"):
                meta.setdefault("og_title", content)
            elif prop in ("og:description", "twitter:description", "description"):
                meta.setdefault("description", content)
            elif prop == "og:site_name":
                meta["site_name"] = content
            elif prop in ("author", "article:author"):
                meta.setdefault("author", content)
            elif prop in ("article:published_time", "datepublished"):
                meta.setdefault("published_at", content)

        for ld in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(ld.string or "")
                if isinstance(data, list):
                    data = data[0]
                if isinstance(data, dict):
                    if "name" in data:
                        meta.setdefault("og_title", str(data["name"]))
                    if "description" in data:
                        meta.setdefault("description", str(data["description"]))
                    if "author" in data:
                        author = data["author"]
                        meta.setdefault(
                            "author",
                            author.get("name", "") if isinstance(author, dict) else str(author),
                        )
                    if "datePublished" in data:
                        meta.setdefault("published_at", str(data["datePublished"]))
            except Exception:
                pass

        return meta


class HTMLSourceResolver:
    MOBILE_UA = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
        "Mobile/15E148 Safari/604.1"
    )

    def __init__(self, timeout: int = 10, allow_network: bool = True):
        self.timeout = timeout
        self.allow_network = allow_network

    def resolve(self, raw_html: str | bytes, url: str = "") -> Tuple[str | bytes, Dict[str, str]]:
        info: Dict[str, str] = {}
        if not url:
            return raw_html, info

        if not self._is_naver_blog_url(url):
            return raw_html, info

        html_text, _ = NormalizeLayer._decode(raw_html) if raw_html else ("", "utf-8")
        soup = (
            NormalizeLayer._parse_with_fallback(html_text)
            if html_text
            else BeautifulSoup("", "html.parser")
        )

        if self._looks_like_naver_post_html(soup):
            info["resolved_strategy"] = "input-already-post-html"
            return html_text, info

        candidates: List[str] = []

        iframe_src = self._find_naver_iframe_src(soup, url)
        if iframe_src:
            candidates.append(iframe_src)

        blog_id, log_no = self._extract_naver_blog_parts(url, soup)
        if blog_id and log_no:
            candidates.extend(
                [
                    f"https://blog.naver.com/PostView.naver?blogId={blog_id}&logNo={log_no}",
                    f"https://m.blog.naver.com/PostView.naver?blogId={blog_id}&logNo={log_no}",
                    f"https://m.blog.naver.com/{blog_id}/{log_no}",
                ]
            )

        deduped: List[str] = []
        seen = set()
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                deduped.append(c)

        if self.allow_network:
            for cand in deduped:
                fetched = self._fetch(cand)
                if not fetched:
                    continue
                fsoup = NormalizeLayer._parse_with_fallback(fetched)
                if self._looks_like_naver_post_html(fsoup):
                    info["resolved_url"] = cand
                    info["resolved_strategy"] = "fetched-naver-post"
                    return fetched, info

        if html_text:
            info["resolved_strategy"] = "fallback-input-shell"
            return html_text, info

        return "", info

    def _is_naver_blog_url(self, url: str) -> bool:
        host = (urlparse(url).netloc or "").lower()
        return host in {"blog.naver.com", "m.blog.naver.com"}

    def _find_naver_iframe_src(self, soup: BeautifulSoup, base_url: str) -> str:
        for tag in soup.find_all(["iframe", "frame"]):
            src = (tag.get("src") or "").strip()
            tag_id = (tag.get("id") or "").strip()
            tag_name = (tag.get("name") or "").strip()

            if not src:
                continue

            if tag_id == "mainFrame" or tag_name == "mainFrame" or "PostView.naver" in src:
                return urljoin(base_url, src)

        raw = str(soup)
        m = re.search(r'src=["\']([^"\']*PostView\.naver[^"\']*)["\']', raw, re.IGNORECASE)
        if m:
            return urljoin(base_url, m.group(1))

        return ""

    def _extract_naver_blog_parts(self, url: str, soup: BeautifulSoup) -> Tuple[str, str]:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)

        blog_id = (qs.get("blogId") or [""])[0]
        log_no = (qs.get("logNo") or [""])[0]

        if blog_id and log_no:
            return blog_id, log_no

        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2 and path_parts[0] != "PostView.naver":
            cand_blog_id = path_parts[0]
            cand_log_no = path_parts[1]
            if re.fullmatch(r"\d+", cand_log_no or ""):
                return cand_blog_id, cand_log_no

        iframe_src = self._find_naver_iframe_src(soup, url)
        if iframe_src:
            p2 = urlparse(iframe_src)
            q2 = parse_qs(p2.query)
            blog_id = (q2.get("blogId") or [""])[0]
            log_no = (q2.get("logNo") or [""])[0]
            if blog_id and log_no:
                return blog_id, log_no

        return "", ""

    def _looks_like_naver_post_html(self, soup: BeautifulSoup) -> bool:
        if not soup:
            return False

        body = soup.body or soup
        text = _normalize_ws(body.get_text(" ", strip=True))
        html = str(soup)

        has_mainframe = (
            'id="mainFrame"' in html
            or "id='mainFrame'" in html
            or 'name="mainFrame"' in html
            or "name='mainFrame'" in html
        )
        has_post_markers = any(
            marker in html
            for marker in [
                "PostView.naver",
                "se-main-container",
                "postViewArea",
                "post-view",
                "se_textarea",
            ]
        )

        if has_mainframe and len(text) < 500:
            return False

        if has_post_markers and len(text) >= 150:
            return True

        if len(text) >= 800 and not has_mainframe:
            return True

        return False

    def _fetch(self, url: str) -> str:
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": self.MOBILE_UA,
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                },
            )
            with urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type", "")
                m = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type, re.IGNORECASE)
                enc = m.group(1) if m else "utf-8"
                try:
                    return data.decode(enc, errors="replace")
                except Exception:
                    return data.decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, ValueError):
            return ""
        except Exception:
            return ""


class NonRenderRemover:
    def clean(self, root: Tag, base_url: str = "") -> None:
        for tag in root.find_all(list(_NONRENDER_TAGS)):
            tag.decompose()

        for c in root.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()

        for tag in root.find_all(attrs={"aria-hidden": "true"}):
            tag.decompose()

        for img in root.find_all("img"):
            alt = _normalize_ws(img.get("alt", ""))
            src = _normalize_ws(img.get("src", ""))

            if src:
                resolved_src = urljoin(base_url, src) if base_url else src
                markdown_img = f"![{alt or 'image'}]({resolved_src})"
                img.replace_with(NavigableString(markdown_img))
            elif alt:
                img.replace_with(NavigableString(f"[이미지: {alt}]"))
            else:
                img.decompose()


class TablePreprocessor:
    def preprocess(self, root: Tag) -> None:
        """
        HTML 문서 안의 table을 RAG용 placeholder text로 교체한다.

        핵심:
        - service layer에서는 outermost parent table만 처리한다.
        - nested child table은 html_table.process_cell() 내부 재귀가 처리한다.
        - root.find_all("table") 전체를 그대로 돌리면 parent/child table이 중복 처리될 수 있다.
        """
        tables = self._find_outermost_tables(root)

        for idx, table in enumerate(tables, start=1):
            body = self._table_to_text(table)
            if not _is_meaningful(body):
                continue

            label = _find_table_caption(table, idx)
            holder = self._make_placeholder(table, label, body, idx)
            table.replace_with(holder)

    def _find_outermost_tables(self, root: Tag) -> list[Tag]:
        """
        현재 root 안에서 최상위 table만 반환한다.
        table 안에 들어 있는 nested table은 여기서 제외한다.
        nested table은 html_table.py의 process_cell()에서 재귀 처리된다.
        """
        tables = []

        for table in root.find_all("table"):
            if table.find_parent("table") is None:
                tables.append(table)

        return tables

    def _table_to_text(self, table: Tag) -> str:
        try:
            import inspect
            import modules.agent_speed.tools.external_extract_main_content.html_table as ht

            def direct_table_children(t: Tag, name: str) -> list[Tag]:
                return [
                    x for x in t.find_all(name)
                    if x.find_parent("table") is t
                ]

            print("HTML_TABLE_FILE:", ht.__file__, flush=True)
            print("PROCESS_FUNC_FILE:", inspect.getsourcefile(ht.process_html_table_be), flush=True)
            print("THEAD_COUNT_IN_SERVICE:", len(table.find_all("thead")), flush=True)
            print("TOP_LEVEL_THEAD_COUNT:", len(direct_table_children(table, "thead")), flush=True)

            if hasattr(ht, "get_header_structure"):
                print("HEADERS_IN_SERVICE:", ht.get_header_structure(None, table), flush=True)
            else:
                print("NO get_header_structure IN html_table", flush=True)

            rows = process_html_table_be(str(table))
            if not rows:
                return ""
            return _normalize_block(json_to_multiline_text(json.dumps(rows, ensure_ascii=False)))
        except Exception as e:
            print("TABLE_PARSE_ERROR:", repr(e), flush=True)
            return ""
    

    def _make_placeholder(self, table: Tag, label: str, body: str, idx: int) -> Tag:
        holder = _new_tag_like(table, "div")
        holder[_TABLE_PLACEHOLDER_ATTR] = "1"
        holder["data-table-index"] = str(idx)

        if label:
            holder["data-table-label"] = label

        holder.append(NavigableString(body))
        return holder

class BodyExtractor:
    def extract(self, soup: BeautifulSoup) -> Tag:
        body = soup.body or soup
        candidates = self._collect_candidates(body)

        if not candidates:
            return body

        best = max(candidates, key=self._rank)
        best = self._drill_down(best)
        region = self._expand_to_region(best)

        if region is not None:
            return region
        return best

    def _collect_candidates(self, body: Tag) -> List[Tag]:
        result: List[Tag] = []
        for node in body.find_all(list(_BLOCK_TAGS)):
            if not isinstance(node, Tag):
                continue
            if _text_len(node) < MIN_CANDIDATE_TEXT:
                continue
            result.append(node)

        if _text_len(body) >= MIN_CANDIDATE_TEXT:
            result.append(body)

        return result

    def _rank(self, node: Tag) -> float:
        score = _node_score(node)
        text_len = _text_len(node)
        text_bonus = min(text_len / 4000.0, 0.25)
        heading_bonus = 0.05 if node.find(["h1", "h2", "h3"]) else 0.0
        p_bonus = min(len(node.find_all("p")) * 0.01, 0.08)

        block_children = [
            c for c in node.children
            if isinstance(c, Tag) and (c.name or "").lower() in _BLOCK_TAGS
        ]
        repeated_bonus = self._repeated_sibling_bonus(block_children)

        return score + text_bonus + heading_bonus + p_bonus + repeated_bonus

    def _drill_down(self, node: Tag) -> Tag:
        cur = node
        for _ in range(4):
            children = [
                c
                for c in cur.children
                if isinstance(c, Tag)
                and (c.name or "").lower() in _BLOCK_TAGS
                and _text_len(c) >= MIN_MEANINGFUL
            ]
            if not children:
                break

            parent_rank = self._rank(cur)
            parent_text = _text_len(cur)
            dense_parent = parent_text / max(len(str(cur)), 1)

            child_rows: List[Tuple[float, Tag]] = []
            for child in children:
                child_rank = self._rank(child)
                child_text = _text_len(child)
                dense_child = child_text / max(len(str(child)), 1)

                descend_ok = (
                    child_rank >= parent_rank * 0.90
                    and child_text >= parent_text * 0.22
                    and dense_child >= dense_parent * 0.95
                )

                if self._looks_like_repeating_container(cur):
                    descend_ok = descend_ok and child_text >= parent_text * 0.35

                if descend_ok:
                    child_rows.append((child_rank, child))

            if not child_rows:
                break

            child_rows.sort(key=lambda x: x[0], reverse=True)
            best_child = child_rows[0][1]

            if len(child_rows) >= 2:
                top1 = child_rows[0][0]
                top2 = child_rows[1][0]
                if top2 >= top1 * 0.94 and self._looks_like_repeating_container(cur):
                    break

            cur = best_child

        return cur

    def _expand_to_region(self, anchor: Tag) -> Optional[Tag]:
        parent = anchor.parent if isinstance(anchor.parent, Tag) else None
        if not parent:
            return None

        siblings = [
            c
            for c in parent.children
            if isinstance(c, Tag)
            and (c.name or "").lower() in _BLOCK_TAGS
            and _text_len(c) >= MIN_MEANINGFUL
        ]
        if len(siblings) < 2:
            return None

        try:
            anchor_idx = siblings.index(anchor)
        except ValueError:
            return None

        left = anchor_idx
        right = anchor_idx

        while left - 1 >= 0 and self._should_merge_sibling(siblings[left - 1], siblings[left], parent):
            left -= 1

        while right + 1 < len(siblings) and self._should_merge_sibling(siblings[right], siblings[right + 1], parent):
            right += 1

        chosen = siblings[left:right + 1]
        if not chosen:
            return None

        chosen_text = sum(_text_len(x) for x in chosen)
        parent_text = _text_len(parent)

        if chosen_text < max(MIN_CANDIDATE_TEXT, int(parent_text * 0.35)):
            return None

        holder = _new_tag_like(parent, "div")
        holder["data-extracted-region"] = "1"
        for node in chosen:
            holder.append(node.extract())
        return holder

    def _should_merge_sibling(self, left: Tag, right: Tag, parent: Tag) -> bool:
        left_name = (left.name or "").lower()
        right_name = (right.name or "").lower()

        if left_name != right_name:
            return False

        left_text = _text_len(left)
        right_text = _text_len(right)
        if left_text < MIN_MEANINGFUL or right_text < MIN_MEANINGFUL:
            return False

        left_rank = self._rank(left)
        right_rank = self._rank(right)

        if min(left_rank, right_rank) < 0.18:
            return False

        if self._same_structure(left, right):
            return True

        if self._looks_like_repeating_container(parent):
            if min(left_rank, right_rank) >= max(left_rank, right_rank) * 0.72:
                return True

        return False

    def _same_structure(self, a: Tag, b: Tag) -> bool:
        if (a.name or "").lower() != (b.name or "").lower():
            return False

        a_children = [c for c in a.children if isinstance(c, Tag)]
        b_children = [c for c in b.children if isinstance(c, Tag)]

        if not a_children or not b_children:
            return False

        a_sig = self._child_signature(a_children)
        b_sig = self._child_signature(b_children)

        if a_sig == b_sig:
            return True

        overlap = 0
        for x, y in zip(a_sig, b_sig):
            if x == y:
                overlap += 1
            else:
                break

        return overlap >= 2

    def _child_signature(self, children: List[Tag]) -> Tuple[str, ...]:
        sig: List[str] = []
        for child in children[:8]:
            name = (child.name or "").lower()
            cls = " ".join(child.get("class", [])) if child.has_attr("class") else ""
            cls = cls.strip().lower()

            if name == "p":
                if "title" in cls or "smalltitle" in cls or "sub-title" in cls:
                    sig.append("p:title")
                elif "desc" in cls or "dot-list" in cls or "line-list" in cls or "tip" in cls:
                    sig.append("p:desc")
                else:
                    sig.append("p")
            elif name == "div":
                if "sub-content" in cls:
                    sig.append("div:sub")
                elif "content-wrap" in cls:
                    sig.append("div:wrap")
                else:
                    sig.append("div")
            else:
                sig.append(name)
        return tuple(sig)

    def _looks_like_repeating_container(self, node: Tag) -> bool:
        children = [
            c
            for c in node.children
            if isinstance(c, Tag)
            and (c.name or "").lower() in _BLOCK_TAGS
            and _text_len(c) >= MIN_MEANINGFUL
        ]
        if len(children) < 2:
            return False

        sig_counts: Dict[Tuple[str, ...], int] = {}
        for child in children:
            sig = self._child_signature([c for c in child.children if isinstance(c, Tag)])
            if not sig:
                continue
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

        if not sig_counts:
            return False

        best = max(sig_counts.values())
        return best >= 2

    def _repeated_sibling_bonus(self, children: List[Tag]) -> float:
        if len(children) < 2:
            return 0.0

        sig_counts: Dict[Tuple[str, ...], int] = {}
        for child in children:
            grand = [c for c in child.children if isinstance(c, Tag)]
            sig = self._child_signature(grand)
            if not sig:
                continue
            sig_counts[sig] = sig_counts.get(sig, 0) + 1

        if not sig_counts:
            return 0.0

        best = max(sig_counts.values())
        ratio = best / max(len(children), 1)
        return min(ratio * 0.12, 0.12)


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CODE = "code"


@dataclass
class DocumentChunk:
    chunk_id: str
    url: str
    title: str
    heading: str
    text: str
    block_type: BlockType
    page: Optional[int] = None
    header_path: Tuple[str, ...] = field(default_factory=tuple)
    header_levels: Tuple[Tuple[int, str], ...] = field(default_factory=tuple)
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
        if not self.header_levels:
            self.header_levels = tuple(_parse_header_levels(self.text))
        if not self.header_path:
            self.header_path = _levels_to_header_path(self.header_levels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "url": self.url,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "block_type": self.block_type.value,
            "page": self.page,
            "header_path": list(self.header_path),
            "header_levels": [list(x) for x in self.header_levels],
            "char_count": self.char_count,
        }


def _parse_header_levels(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if not m:
            continue
        out.append((len(m.group(1)), m.group(2).strip()))
    return out


def _levels_to_header_path(levels: List[Tuple[int, str]] | Tuple[Tuple[int, str], ...]) -> Tuple[str, ...]:
    path: List[str] = []
    for level, header in levels:
        while len(path) >= level:
            path.pop()
        path.append(header)
    return tuple(path)


def _header_levels_from_splitter_metadata(meta: Dict[str, Any]) -> Tuple[Tuple[int, str], ...]:
    levels: List[Tuple[int, str]] = []
    if not isinstance(meta, dict):
        return tuple(levels)

    for level in range(1, 7):
        value = meta.get(f"h{level}")
        value = _normalize_ws(str(value or ""))
        if value:
            levels.append((level, value))

    return tuple(levels)


@dataclass
class ExtractionResult:
    url: str
    title: str
    meta: Dict[str, str]
    chunks: List[DocumentChunk]
    ok: bool
    error: Optional[str] = None

    @property
    def total_chars(self) -> int:
        return sum(c.char_count for c in self.chunks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "meta": self.meta,
            "chunks": [c.to_dict() for c in self.chunks],
            "ok": self.ok,
            "error": self.error,
            "total_chars": self.total_chars,
        }


def _resolve_fallback_title(title: str, url: str) -> str:
    title = _normalize_ws(title)
    if title:
        return unquote(title)

    if url:
        path = url.rstrip("/").rsplit("/", 1)[-1]
        guessed = re.sub(r"[_\-]", " ", unquote(path)).split("?")[0].strip()
        if guessed:
            return guessed

    return ""


def _is_explicit_pdf_content_type(doc: Dict[str, Any]) -> bool:
    content_type = str(doc.get("content_type", "") or "").lower().strip()
    source_content_type = str(doc.get("source_content_type", "") or "").lower().strip()
    return (
        content_type == "pdf"
        or "application/pdf" in content_type
        or "application/pdf" in source_content_type
    )


def _looks_like_pdf_document(doc: Dict[str, Any]) -> bool:
    if not isinstance(doc, dict):
        return False

    if _is_explicit_pdf_content_type(doc):
        return True

    url = str(doc.get("url", "") or "")
    final_url = str(doc.get("final_url", "") or "")
    title = str(doc.get("title", "") or "")
    heading = str(doc.get("heading", "") or "")
    search_title = str(doc.get("search_title", "") or "")
    source_content_type = str(doc.get("source_content_type", "") or "")

    low_joined = " ".join([url, final_url, title, heading, search_title, source_content_type]).lower()
    if ".pdf" in low_joined:
        return True

    blocks = doc.get("blocks")
    if isinstance(blocks, list) and blocks:
        first = blocks[0]
        if isinstance(first, dict):
            block_ct = str(first.get("content_type", "") or "").lower()
            if "pdf" in block_ct:
                return True

    return False


def _is_preextracted_document(doc: Dict[str, Any]) -> bool:
    if not isinstance(doc, dict):
        return False

    blocks = doc.get("blocks")
    text = doc.get("text")
    raw_text = doc.get("raw_text")
    raw_html = doc.get("raw_html")
    content_type = str(doc.get("content_type", "") or "").lower().strip()

    if isinstance(blocks, list) and len(blocks) > 0:
        return True

    if text and not _looks_like_html_markup(str(text)):
        return True

    if raw_text and not raw_html:
        return True

    if content_type in {"text", "txt", "text/plain", "docx", "hwp", "hwpx"}:
        return True

    return False


def _pdf_heading_from_block(block: Dict[str, Any], page: Optional[int], title: str) -> str:
    heading = _normalize_ws(str(block.get("heading", "") or ""))
    if heading:
        return heading
    if page is not None:
        return f"Page {page}"
    return title


def _render_pdf_chunk_text(body: str, heading: str) -> str:
    body = _normalize_block(body)
    if not body:
        return ""
    heading = _normalize_ws(heading)
    if heading and not re.match(r"^\s*#{1,6}\s+", body):
        return f"# {heading}\n\n{body}"
    return body


def _make_pdf_subchunk_id(
    *,
    base_chunk_id: str,
    url: str,
    heading: str,
    page: Optional[int],
    sub_idx: int,
    text: str,
) -> str:
    if base_chunk_id:
        return base_chunk_id if sub_idx == 0 else f"{base_chunk_id}::{sub_idx:03d}"

    h = hashlib.md5(
        f"{url}|{heading}|{page}|{sub_idx}|{text[:64]}".encode()
    ).hexdigest()[:8]
    return f"pdf_{sub_idx:04d}_{h}"


def _pdf_split_paragraphs(text: str) -> List[str]:
    text = _normalize_block(text)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts


def _merge_short_pieces(parts: List[str], target: int = TARGET_CHUNK_CHARS) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            out.append(_normalize_block("\n\n".join(cur)))
        cur = []
        cur_len = 0

    for part in parts:
        part = _normalize_block(part)
        if not part:
            continue

        projected = cur_len + len(part) + (2 if cur else 0)
        if cur and projected > target:
            flush()

        cur.append(part)
        cur_len += len(part) + (2 if len(cur) > 1 else 0)

        if cur_len >= target:
            flush()

    flush()
    return [x for x in out if x]


def _split_pdf_body_force(text: str) -> List[str]:
    text = _normalize_block(text)
    if not text:
        return []

    if len(text) <= TARGET_CHUNK_CHARS:
        return [text]

    paragraphs = _pdf_split_paragraphs(text)
    if len(paragraphs) > 1:
        merged = _merge_short_pieces(paragraphs, TARGET_CHUNK_CHARS)
        if merged:
            final_parts: List[str] = []
            for piece in merged:
                if len(piece) <= MAX_CHUNK_CHARS:
                    final_parts.append(piece)
                else:
                    final_parts.extend(_hard_slice(piece, TARGET_CHUNK_CHARS))
            return [x for x in final_parts if x]

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        merged = _merge_short_pieces(sentences, TARGET_CHUNK_CHARS)
        if merged:
            final_parts: List[str] = []
            for piece in merged:
                if len(piece) <= MAX_CHUNK_CHARS:
                    final_parts.append(piece)
                else:
                    final_parts.extend(_hard_slice(piece, TARGET_CHUNK_CHARS))
            return [x for x in final_parts if x]

    return _hard_slice(text, TARGET_CHUNK_CHARS)


def _chunk_pdf_text_to_document_chunks(
    *,
    url: str,
    title: str,
    heading: str,
    body: str,
    page: Optional[int],
    base_chunk_id: str = "",
) -> List[DocumentChunk]:
    body = _normalize_block(body)
    if not body:
        return []

    effective_heading = _normalize_ws(heading or title)
    parts = _split_pdf_body_force(body)

    out: List[DocumentChunk] = []

    for sub_idx, part in enumerate(parts):
        part = _normalize_block(part)
        if not part:
            continue

        text = _render_pdf_chunk_text(part, effective_heading)
        if not text:
            continue

        chunk_id = _make_pdf_subchunk_id(
            base_chunk_id=base_chunk_id,
            url=url,
            heading=effective_heading,
            page=page,
            sub_idx=sub_idx,
            text=text,
        )

        out.append(
            DocumentChunk(
                chunk_id=chunk_id,
                url=url,
                title=title,
                heading=effective_heading,
                text=text,
                block_type=BlockType.TEXT,
                page=page,
            )
        )

    return out


def _build_pdf_chunks_from_doc(doc: Dict[str, Any]) -> List[DocumentChunk]:
    url = str(doc.get("url", "") or "")
    title = _resolve_fallback_title(
        str(doc.get("title", "") or doc.get("search_title", "") or ""),
        url,
    )

    out: List[DocumentChunk] = []

    block_id = str(doc.get("block_id") or "")
    doc_text = _normalize_block(str(doc.get("text", "") or ""))
    doc_heading = _normalize_ws(str(doc.get("heading", "") or ""))
    doc_page = doc.get("page")
    try:
        doc_page = int(doc_page) if doc_page is not None else None
    except Exception:
        doc_page = None

    if block_id and doc_text and not doc.get("blocks"):
        out.extend(
            _chunk_pdf_text_to_document_chunks(
                url=url,
                title=title,
                heading=doc_heading or title,
                body=doc_text,
                page=doc_page,
                base_chunk_id=block_id,
            )
        )
        if out:
            return out

    raw_blocks = doc.get("blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        for idx, block in enumerate(raw_blocks):
            if not isinstance(block, dict):
                continue

            block_text = _normalize_block(str(block.get("text", "") or ""))
            if not block_text:
                continue

            page = block.get("page")
            try:
                page = int(page) if page is not None else None
            except Exception:
                page = None

            block_heading = _normalize_ws(str(block.get("heading", "") or ""))
            block_url = str(block.get("url", "") or url)
            block_title = _resolve_fallback_title(
                str(block.get("title", "") or title),
                block_url or url,
            )
            base_heading = block_heading or _pdf_heading_from_block(block, page, block_title)
            base_chunk_id = str(block.get("block_id") or f"pdf_block_{idx:04d}")

            out.extend(
                _chunk_pdf_text_to_document_chunks(
                    url=block_url,
                    title=block_title,
                    heading=base_heading,
                    body=block_text,
                    page=page,
                    base_chunk_id=base_chunk_id,
                )
            )

        if out:
            return out

    raw_chunks = doc.get("chunks")
    if isinstance(raw_chunks, list) and raw_chunks:
        for idx, item in enumerate(raw_chunks):
            if isinstance(item, dict):
                item_text = _normalize_block(str(item.get("text", "") or ""))
                if not item_text:
                    continue
                page = item.get("page")
                try:
                    page = int(page) if page is not None else None
                except Exception:
                    page = None
                item_heading = _normalize_ws(str(item.get("heading", "") or title))
                out.extend(
                    _chunk_pdf_text_to_document_chunks(
                        url=str(item.get("url", "") or url),
                        title=_resolve_fallback_title(str(item.get("title", "") or title), url),
                        heading=item_heading or title,
                        body=item_text,
                        page=page,
                        base_chunk_id=str(item.get("block_id") or f"raw_chunk_{idx:04d}"),
                    )
                )
            else:
                text = _normalize_block(str(item or ""))
                if not text:
                    continue
                out.extend(
                    _chunk_pdf_text_to_document_chunks(
                        url=url,
                        title=title,
                        heading=title,
                        body=text,
                        page=None,
                        base_chunk_id=f"raw_chunk_{idx:04d}",
                    )
                )

        if out:
            return out

    fallback_text = _normalize_block(
        str(doc.get("raw_text", "") or doc.get("text", "") or "")
    )
    if fallback_text:
        out.extend(
            _chunk_pdf_text_to_document_chunks(
                url=url,
                title=title,
                heading=title,
                body=fallback_text,
                page=None,
                base_chunk_id="raw_text",
            )
        )

    if not out and title:
        text = f"# {title}"
        out.append(
            DocumentChunk(
                chunk_id=_make_chunk_id(url, 0, text),
                url=url,
                title=title,
                heading=title,
                text=text,
                block_type=BlockType.TEXT,
                page=None,
            )
        )

    return out


def _pdf_doc_to_extraction_result(
    doc: Dict[str, Any],
    debug_print=None,
) -> ExtractionResult:
    url = str(doc.get("url", "") or "")
    title = _resolve_fallback_title(
        str(doc.get("title", "") or doc.get("search_title", "") or ""),
        url,
    )
    chunks = _build_pdf_chunks_from_doc(doc)

    meta: Dict[str, str] = {
        "content_type": "pdf",
    }
    if doc.get("final_url"):
        meta["final_url"] = str(doc.get("final_url"))
    if doc.get("search_title"):
        meta["search_title"] = _normalize_ws(str(doc.get("search_title")))
    if doc.get("display_link"):
        meta["display_link"] = _normalize_ws(str(doc.get("display_link")))
    if doc.get("source_engine"):
        meta["source_engine"] = _normalize_ws(str(doc.get("source_engine")))
    if doc.get("page_count") is not None:
        meta["page_count"] = str(doc.get("page_count"))

    if callable(debug_print):
        try:
            debug_print(
                "PDF CHUNK BUILT",
                {
                    "url": url,
                    "title": title,
                    "chunk_count": len(chunks),
                    "chunk_ids": [c.chunk_id for c in chunks[:20]],
                    "chunk_sizes": [c.char_count for c in chunks[:20]],
                    "pages": [c.page for c in chunks[:20]],
                    "meta": meta,
                },
            )
        except Exception:
            pass

    return ExtractionResult(
        url=url,
        title=title,
        meta=meta,
        chunks=chunks,
        ok=True,
    )


def _doc_to_extraction_result_from_preextracted(
    doc: Dict[str, Any],
    debug_print=None,
) -> ExtractionResult:
    url = str(doc.get("url", "") or "")
    title = _resolve_fallback_title(
        str(doc.get("title", "") or doc.get("search_title", "") or ""),
        url,
    )
    content_type = str(doc.get("content_type", "") or "unknown").lower().strip()

    chunks: List[DocumentChunk] = []
    idx = 0

    raw_blocks = doc.get("blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue

            block_text = _normalize_block(str(block.get("text", "") or ""))
            if not block_text:
                continue

            block_heading = _normalize_ws(str(block.get("heading", "") or "")) or title
            block_url = str(block.get("url", "") or url)
            block_title = _resolve_fallback_title(
                str(block.get("title", "") or title),
                block_url or url,
            )
            page = block.get("page")
            try:
                page = int(page) if page is not None else None
            except Exception:
                page = None

            rendered_text = block_text
            if block_heading and not re.match(r"^\s*#{1,6}\s+", rendered_text):
                rendered_text = f"# {block_heading}\n\n{rendered_text}"

            chunks.append(
                DocumentChunk(
                    chunk_id=str(block.get("block_id") or _make_chunk_id(block_url, idx, rendered_text)),
                    url=block_url,
                    title=block_title,
                    heading=block_heading,
                    text=rendered_text,
                    block_type=BlockType.TEXT,
                    page=page,
                )
            )
            idx += 1

    if not chunks:
        source_text = _normalize_block(
            str(doc.get("text", "") or doc.get("raw_text", "") or "")
        )
        if source_text:
            heading = title
            parts = _hard_slice(source_text, TARGET_CHUNK_CHARS)
            for part in parts:
                part = _normalize_block(part)
                if not part:
                    continue
                rendered = part
                if heading and not re.match(r"^\s*#{1,6}\s+", rendered):
                    rendered = f"# {heading}\n\n{rendered}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=_make_chunk_id(url, idx, rendered),
                        url=url,
                        title=title,
                        heading=heading,
                        text=rendered,
                        block_type=BlockType.TEXT,
                        page=None,
                    )
                )
                idx += 1

    meta: Dict[str, str] = {
        "content_type": content_type or "unknown",
        "chunker": "preextracted_blocks_or_text",
        "chunk_target_chars": str(TARGET_CHUNK_CHARS),
        "chunk_max_chars": str(MAX_CHUNK_CHARS),
    }
    if doc.get("final_url"):
        meta["final_url"] = str(doc.get("final_url"))
    if doc.get("search_title"):
        meta["search_title"] = _normalize_ws(str(doc.get("search_title")))
    if doc.get("display_link"):
        meta["display_link"] = _normalize_ws(str(doc.get("display_link")))
    if doc.get("source_engine"):
        meta["source_engine"] = _normalize_ws(str(doc.get("source_engine")))

    if callable(debug_print):
        try:
            debug_print(
                "PREEXTRACTED CHUNK BUILT",
                {
                    "url": url,
                    "title": title,
                    "content_type": content_type,
                    "chunk_count": len(chunks),
                    "chunk_ids": [c.chunk_id for c in chunks[:20]],
                    "chunk_sizes": [c.char_count for c in chunks[:20]],
                },
            )
        except Exception:
            pass

    return ExtractionResult(
        url=url,
        title=title,
        meta=meta,
        chunks=chunks,
        ok=True,
    )


class SemanticChunker:
    def __init__(self):
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
            return_each_line=False,
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TARGET_CHUNK_CHARS,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, root: Tag, *, url: str, title: str) -> List[DocumentChunk]:
        effective_title = _normalize_ws(title)
        markdown_text = self._render_markdown(root, effective_title)
        markdown_text = _normalize_block(markdown_text)

        if not markdown_text:
            return []

        header_docs = self.header_splitter.split_text(markdown_text)

        raw_chunks: List[DocumentChunk] = []
        idx = 0

        for doc in header_docs:
            section_text = _normalize_block(doc.page_content)
            if not section_text:
                continue

            splitter_levels = _header_levels_from_splitter_metadata(getattr(doc, "metadata", {}) or {})
            splitter_path = _levels_to_header_path(splitter_levels)
            split_parts = self._split_section(section_text)

            for part in split_parts:
                part = _normalize_block(part)
                if not _is_meaningful(_strip_markdown_headings(part)):
                    continue

                raw_chunks.append(
                    DocumentChunk(
                        chunk_id=_make_chunk_id(url, idx, part),
                        url=url,
                        title=effective_title,
                        heading=self._extract_last_heading(part, effective_title),
                        text=part,
                        block_type=self._infer_block_type(part),
                        page=None,
                        header_path=splitter_path,
                        header_levels=splitter_levels,
                    )
                )
                idx += 1

        merged = self._merge_small_same_path(raw_chunks, url=url, title=effective_title)
        merged = self._merge_small_sibling_sections(merged, url=url, title=effective_title)
        return merged

    def _render_markdown(self, root: Tag, title: str) -> str:
        body = root.body if isinstance(root, BeautifulSoup) else None
        start = body or root
        lines: List[str] = []

        def walk(node: Any):
            if isinstance(node, NavigableString):
                return
            if not isinstance(node, Tag):
                return

            name = (node.name or "").lower()
            if name in _NONRENDER_TAGS:
                return

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                text = _normalize_ws(node.get_text(" ", strip=True))
                if text:
                    lines.append(f"{'#' * level} {text}")
                    lines.append("")
                return

            if _is_preprocessed_table(node):
                label = _normalize_ws(node.get("data-table-label", "") or "table")
                body_text = _normalize_block(node.get_text("\n", strip=True))
                if label:
                    lines.append(f"### {label}")
                    lines.append("")
                if body_text:
                    lines.append(body_text)
                    lines.append("")
                return

            if name in ("pre", "code"):
                code_text = _normalize_code(node.get_text("\n", strip=False))
                if _is_meaningful(code_text):
                    lines.append("```")
                    lines.append(code_text)
                    lines.append("```")
                    lines.append("")
                return

            if name in ("p", "blockquote", "li"):
                text = _extract_block_text(node)
                if _is_meaningful(text):
                    if name == "blockquote":
                        lines.append(f"> {text}")
                    elif name == "li":
                        lines.append(f"- {text}")
                    else:
                        lines.append(text)
                    lines.append("")
                return

            if name in ("ul", "ol"):
                items: List[str] = []
                for li in node.find_all("li", recursive=False):
                    item = _extract_block_text(li)
                    if item:
                        items.append(f"- {item}")
                if items:
                    lines.extend(items)
                    lines.append("")
                return

            direct_block_children = [
                c for c in node.children
                if isinstance(c, Tag) and (c.name or "").lower() in _BLOCK_TAGS
            ]
            if direct_block_children:
                for child in direct_block_children:
                    walk(child)
                return

            text = _extract_block_text(node)
            if _is_meaningful(text):
                lines.append(text)
                lines.append("")

        walk(start)
        markdown_text = _normalize_block("\n".join(lines))

        if title and not re.search(r"^#\s+.+$", markdown_text, re.M):
            return f"# {title}\n\n{markdown_text}" if markdown_text else f"# {title}"
        return markdown_text

    def _split_section(self, section_text: str) -> List[str]:
        section_text = _normalize_block(section_text)

        if len(section_text) <= MAX_CHUNK_CHARS:
            return [section_text]

        heading_lines: List[str] = []
        body_lines: List[str] = []
        in_heading = True

        for line in section_text.splitlines():
            if in_heading and re.match(r"^#{1,6}\s+.+$", line):
                heading_lines.append(line)
            else:
                in_heading = False
                body_lines.append(line)

        head = _normalize_block("\n".join(heading_lines))
        body = _normalize_block("\n".join(body_lines))

        if not body:
            return _hard_slice(section_text, TARGET_CHUNK_CHARS)

        body_parts = self.recursive_splitter.split_text(body)

        out: List[str] = []
        for body_part in body_parts:
            body_part = _normalize_block(body_part)
            if not body_part:
                continue

            merged = f"{head}\n\n{body_part}" if head else body_part
            if len(merged) <= MAX_CHUNK_CHARS:
                out.append(merged)
            else:
                out.extend(_hard_slice(merged, TARGET_CHUNK_CHARS))

        return out

    def _extract_last_heading(self, text: str, fallback: str = "") -> str:
        headers = []
        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                headers.append(m.group(2).strip())
        return headers[-1] if headers else fallback

    def _chunk_header_levels(self, chunk: DocumentChunk) -> Tuple[Tuple[int, str], ...]:
        if chunk.header_levels:
            return chunk.header_levels
        return tuple(_parse_header_levels(chunk.text))

    def _chunk_header_path(self, chunk: DocumentChunk) -> Tuple[str, ...]:
        if chunk.header_path:
            return chunk.header_path
        return _levels_to_header_path(self._chunk_header_levels(chunk))

    def _parent_section_key(self, chunk: DocumentChunk) -> Tuple[str, ...]:
        headers = self._chunk_header_levels(chunk)
        if len(headers) <= 1:
            return tuple(f"{lvl}:{txt}" for lvl, txt in headers)
        return tuple(f"{lvl}:{txt}" for lvl, txt in headers[:-1])

    def _shared_ancestor_depth(self, left: DocumentChunk, right: DocumentChunk) -> int:
        left_headers = self._chunk_header_levels(left)
        right_headers = self._chunk_header_levels(right)
        depth = 0
        for (l_lvl, l_txt), (r_lvl, r_txt) in zip(left_headers, right_headers):
            if l_lvl != r_lvl or l_txt != r_txt:
                break
            depth += 1
        return depth

    def _infer_block_type(self, text: str) -> BlockType:
        t = _normalize_block(text)
        if "```" in t:
            return BlockType.CODE
        if re.search(r"^\|\s*.+\s*\|$", t, re.M):
            return BlockType.TABLE
        return BlockType.TEXT

    def _merge_small_same_path(
        self,
        chunks: List[DocumentChunk],
        *,
        url: str,
        title: str,
    ) -> List[DocumentChunk]:
        if not chunks:
            return []

        merged: List[DocumentChunk] = []

        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            prev = merged[-1]
            prev_path = self._chunk_header_path(prev)
            cur_path = self._chunk_header_path(chunk)

            same_path = prev_path == cur_path
            projected = len(prev.text) + 2 + len(_strip_markdown_headings(chunk.text))

            if (
                same_path
                and prev.block_type == chunk.block_type == BlockType.TEXT
                and len(prev.text) < TARGET_CHUNK_CHARS
                and projected <= MAX_CHUNK_CHARS
            ):
                prev_head_lines = []
                prev_body_lines = []
                head_done = False

                for line in prev.text.splitlines():
                    if not head_done and re.match(r"^#{1,6}\s+.+$", line):
                        prev_head_lines.append(line)
                    else:
                        head_done = True
                        prev_body_lines.append(line)

                combined_head = _normalize_block("\n".join(prev_head_lines))
                combined_body = _normalize_block(
                    "\n\n".join([
                        _normalize_block("\n".join(prev_body_lines)),
                        _strip_markdown_headings(chunk.text),
                    ])
                )

                combined_text = (
                    f"{combined_head}\n\n{combined_body}"
                    if combined_head and combined_body
                    else combined_head or combined_body
                )
                combined_text = _normalize_block(combined_text)

                merged[-1] = DocumentChunk(
                    chunk_id=_make_chunk_id(url, len(merged) - 1, combined_text),
                    url=url,
                    title=title,
                    heading=self._extract_last_heading(combined_text, title),
                    text=combined_text,
                    block_type=prev.block_type,
                    page=prev.page,
                    header_path=prev.header_path,
                    header_levels=prev.header_levels,
                )
            else:
                merged.append(chunk)

        return merged

    def _can_merge_sibling_sections(
        self,
        left: DocumentChunk,
        right: DocumentChunk,
    ) -> bool:
        if left.block_type != BlockType.TEXT or right.block_type != BlockType.TEXT:
            return False

        if self._chunk_header_path(left) == self._chunk_header_path(right):
            return False

        left_len = len(left.text)
        right_len = len(right.text)

        if self._parent_section_key(left) == self._parent_section_key(right):
            return True

        shared_depth = self._shared_ancestor_depth(left, right)
        if shared_depth <= 0:
            return False

        if max(left_len, right_len) > TARGET_CHUNK_CHARS:
            return False

        return True

    def _merge_two_chunks_keep_both_headers(
        self,
        left: DocumentChunk,
        right: DocumentChunk,
        *,
        url: str,
        title: str,
        idx: int,
    ) -> DocumentChunk:
        left_text = _normalize_block(left.text)
        right_text = _normalize_block(right.text)
        merged_text = _normalize_block(f"{left_text}\n\n{right_text}")

        return DocumentChunk(
            chunk_id=_make_chunk_id(url, idx, merged_text),
            url=url,
            title=title,
            heading=left.heading,
            text=merged_text,
            block_type=BlockType.TEXT,
            page=left.page,
            header_path=left.header_path,
            header_levels=left.header_levels,
        )

    def _merge_small_sibling_sections(
        self,
        chunks: List[DocumentChunk],
        *,
        url: str,
        title: str,
    ) -> List[DocumentChunk]:
        if not chunks:
            return []

        merged: List[DocumentChunk] = []
        cur = chunks[0]

        for next_chunk in chunks[1:]:
            cur_len = len(cur.text)
            next_len = len(next_chunk.text)
            projected = cur_len + 2 + next_len

            should_merge = (
                self._can_merge_sibling_sections(cur, next_chunk)
                and max(cur_len, next_len) <= TARGET_CHUNK_CHARS
                and projected <= MAX_CHUNK_CHARS
            )

            if should_merge:
                cur = self._merge_two_chunks_keep_both_headers(
                    cur,
                    next_chunk,
                    url=url,
                    title=title,
                    idx=len(merged),
                )
            else:
                merged.append(cur)
                cur = next_chunk

        merged.append(cur)
        return merged


class HTMLPreprocessor:
    def __init__(self):
        self._source_resolver = HTMLSourceResolver()
        self._cleaner = NonRenderRemover()
        self._table_preprocessor = TablePreprocessor()
        self._extractor = BodyExtractor()
        self._chunker = SemanticChunker()

    def process(self, raw_html: str | bytes, url: str = "", title: str = "") -> ExtractionResult:
        try:
            return self._run(raw_html, url, title)
        except Exception as e:
            return ExtractionResult(
                url=url,
                title=title,
                meta={},
                chunks=[],
                ok=False,
                error=str(e),
            )

    def process_batch(self, docs: List[Dict[str, Any]]) -> List[ExtractionResult]:
        out: List[ExtractionResult] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            out.append(_process_single_document(doc, self))
        return out

    def _resolve_title(self, meta: Dict[str, str], root: Tag, title: str, url: str) -> str:
        if title:
            return _normalize_ws(title)
        if meta.get("og_title"):
            return _normalize_ws(meta["og_title"])
        if meta.get("title"):
            return _normalize_ws(meta["title"])

        for h in root.find_all(["h1", "h2", "h3"]):
            t = _normalize_ws(h.get_text(" ", strip=True))
            if t:
                return t

        if url:
            path = url.rstrip("/").rsplit("/", 1)[-1]
            return re.sub(r"[_\-]", " ", unquote(path)).split("?")[0].strip() or url

        return ""

    def _run(self, raw_html: str | bytes, url: str, title: str) -> ExtractionResult:
        resolved_html, resolve_info = self._source_resolver.resolve(raw_html, url)
        soup, _ = NormalizeLayer.parse(resolved_html, url)
        meta = NormalizeLayer.extract_meta(soup)
        meta.update(resolve_info)

        working = BeautifulSoup(str(soup), "html.parser")
        root = working.body or working

        self._cleaner.clean(root, base_url=url)
        self._table_preprocessor.preprocess(root)

        body_node = self._extractor.extract(working)
        final_title = self._resolve_title(meta, body_node, title, url)
        chunks = self._chunker.chunk(body_node, url=url, title=final_title)

        if not chunks:
            fallback = _normalize_block((body_node or working).get_text("\n", strip=True))
            if _is_meaningful(fallback):
                text = f"# {final_title}\n\n{fallback}" if final_title else fallback
                chunks = [
                    DocumentChunk(
                        chunk_id=_make_chunk_id(url, 0, text),
                        url=url,
                        title=final_title,
                        heading=final_title,
                        text=text,
                        block_type=BlockType.TEXT,
                    )
                ]

        meta["chunker"] = "langchain_markdown_header_plus_sibling_merge"
        meta["chunk_target_chars"] = str(TARGET_CHUNK_CHARS)
        meta["chunk_max_chars"] = str(MAX_CHUNK_CHARS)
        meta.setdefault("content_type", "html")

        return ExtractionResult(
            url=url,
            title=final_title,
            meta=meta,
            chunks=chunks,
            ok=True,
        )


def _result_to_output_document(
    result: ExtractionResult,
    doc: Dict[str, Any],
    *,
    content_type: str,
    debug_print=None,
) -> Dict[str, Any]:
    resolved_content_type = (
        content_type
        or str(result.meta.get("content_type", "") or "")
        or str(doc.get("content_type", "") or "")
        or "unknown"
    )

    blocks = [
        {
            "url": c.url,
            "final_url": doc.get("final_url", result.url),
            "title": c.title,
            "content_type": resolved_content_type,
            "block_id": c.chunk_id,
            "heading": c.heading,
            "text": c.text,
            "page": c.page,
            "search_title": _normalize_ws(str(doc.get("search_title", "") or "")),
            "search_snippet": _normalize_ws(str(doc.get("search_snippet", "") or "")),
            "display_link": _normalize_ws(str(doc.get("display_link", "") or "")),
            "source_engine": _normalize_ws(str(doc.get("source_engine", "") or "")),
        }
        for c in result.chunks
        if _is_meaningful(c.text)
    ]

    output_doc = {
        "url": result.url,
        "final_url": doc.get("final_url", result.url),
        "title": result.title,
        "content_type": resolved_content_type,
        "text": "",
        "content": "",
        "raw_text": "",
        "raw_html": "",
        "chunks": [],
        "links": [],
        "blocks": blocks,
        "meta": result.meta,
        "chunk_count": len(blocks),
        "total_chars": sum(len(str(b.get("text", "") or "")) for b in blocks),
        "search_title": _normalize_ws(str(doc.get("search_title", "") or "")),
        "search_snippet": _normalize_ws(str(doc.get("search_snippet", "") or "")),
        "display_link": _normalize_ws(str(doc.get("display_link", "") or "")),
        "source_engine": _normalize_ws(str(doc.get("source_engine", "") or "")),
    }

    if callable(debug_print):
        try:
            debug_print(
                "OUTPUT DOCUMENT BUILT",
                {
                    "url": output_doc["url"],
                    "title": output_doc["title"],
                    "content_type": output_doc["content_type"],
                    "chunk_count": output_doc["chunk_count"],
                    "block_count": len(output_doc["blocks"]),
                    "block_ids": [b.get("block_id") for b in output_doc["blocks"][:20]],
                    "block_sizes": [len(str(b.get("text", "") or "")) for b in output_doc["blocks"][:20]],
                },
            )
        except Exception:
            pass

    return output_doc


def _process_single_document(
    doc: Dict[str, Any],
    preprocessor: "HTMLPreprocessor",
    debug_print=None,
) -> ExtractionResult:
    content_type = str(doc.get("content_type", "") or "").lower().strip()

    if content_type == "pdf" or "application/pdf" in content_type:
        return _pdf_doc_to_extraction_result(doc, debug_print=debug_print)

    if _looks_like_pdf_document(doc):
        return _pdf_doc_to_extraction_result(doc, debug_print=debug_print)

    if _is_preextracted_document(doc):
        return _doc_to_extraction_result_from_preextracted(doc, debug_print=debug_print)

    return preprocessor.process(
        raw_html=doc.get("raw_html") or "",
        url=doc.get("url", ""),
        title=doc.get("title", ""),
    )


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Dict[str, Any]

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


_DEFAULT_PREPROCESSOR = HTMLPreprocessor()


def run(query: str = "", documents=None, debug_print=None) -> ToolResult:
    if not isinstance(documents, list) or not documents:
        return ToolResult(
            name="external_extract_main_content",
            ok=False,
            data={"error": "documents required"},
        )

    out = []

    for doc in documents:
        if not isinstance(doc, dict):
            continue

        is_pdf = _is_explicit_pdf_content_type(doc) or _looks_like_pdf_document(doc)

        result = _process_single_document(
            doc,
            _DEFAULT_PREPROCESSOR,
            debug_print=debug_print,
        )

        if callable(debug_print):
            try:
                debug_print(
                    "EXTRACTION RESULT",
                    {
                        "url": result.url,
                        "title": result.title,
                        "chunk_count": len(result.chunks),
                        "meta": result.meta,
                        "ok": result.ok,
                        "error": result.error,
                        "is_pdf_path": is_pdf,
                    }
                )
            except Exception:
                pass

        resolved_content_type = (
            "pdf"
            if is_pdf
            else str(result.meta.get("content_type", "") or doc.get("content_type", "") or "unknown")
        )

        output_doc = _result_to_output_document(
            result,
            doc,
            content_type=resolved_content_type,
            debug_print=debug_print,
        )

        out.append(output_doc)

    return ToolResult(
        name="external_extract_main_content",
        ok=True,
        data={"query": query, "documents": out},
    )
