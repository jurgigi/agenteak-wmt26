"""Text normalisation, sentence splitting, chunking and reconstruction.

No model dependencies: this module is pure Python and can be imported and
tested without torch or langgraph installed.

The working unit throughout the pipeline is a CHUNK: at most `max_sentences`
sentences of one paragraph. A very short paragraph ("Uso", "Grados") is a
heading rather than prose; optionally it is carried into the following chunk as
a LEAD and split back out at reconstruction, so the source's line-break
structure survives intact.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def norm(s: str) -> str:
    """Lowercase and strip diacritics. Used for all matching."""
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower())
                   if unicodedata.category(c) != "Mn").strip()


def words(s: str) -> list:
    return re.findall(r"\w[\w-]*", norm(s))


def word_bounded(needle: str, haystack: str) -> bool:
    return bool(needle) and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)",
                                      haystack) is not None


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def between(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    """Content between sentinels; tolerates a missing closing tag."""
    i = text.find(open_tag)
    if i == -1:
        return None
    i += len(open_tag)
    j = text.find(close_tag, i)
    return (text[i:j] if j != -1 else text[i:]).strip()


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d[\d.,]*")


def number_multiset(text: str) -> Counter:
    """Digit sequences with thousands separators removed, so that 15,000 in a
    Spanish source and 15.000 in a Basque translation compare equal."""
    out = Counter()
    for m in _NUM_RE.finditer(text or ""):
        tok = re.sub(r"[.,](?=\d{3}(?!\d))", "", m.group(0).rstrip(".,"))
        tok = tok.replace(",", ".")
        if tok:
            out[tok] += 1
    return out


def number_mismatch(source: str, translation: str) -> dict:
    src, tgt = number_multiset(source), number_multiset(translation)
    missing = sorted((src - tgt).elements())
    added = sorted((tgt - src).elements())
    return {"missing": missing, "added": added, "ok": not missing and not added}


# ---------------------------------------------------------------------------
# Sentence splitting and chunking
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(
    r'(?<=[.!?])["»\')\]]*\s+(?=[¿¡"«(\[A-ZÁÉÍÓÚÜÑ0-9])')

_ABBREV = r"\b(Sr|Sra|Dr|Dra|art|fig|núm|nº|pág|ref|máx|mín|aprox|etc)\."


def split_sentences(paragraph: str) -> list:
    """Regex sentence splitter, guarded for abbreviations common in automotive
    and energy prose."""
    text = re.sub(r"[ \t]+", " ", (paragraph or "").strip())
    if not text:
        return []
    protected = re.sub(_ABBREV, lambda m: m.group(0)[:-1] + "\x00", text,
                       flags=re.IGNORECASE)
    sents = [s.replace("\x00", ".").strip() for s in _SENT_SPLIT.split(protected)]
    return [s for s in sents if s]


def chunk_paragraph(paragraph: str, max_sentences: int = 3,
                    max_chars: int = 1200) -> list:
    """Group sentences into runs of at most `max_sentences`, breaking early if a
    run would exceed `max_chars`."""
    sents = split_sentences(paragraph)
    if not sents:
        return []
    chunks, buf = [], []
    for s in sents:
        trial = buf + [s]
        if buf and (len(trial) > max_sentences
                    or len(" ".join(trial)) > max_chars):
            chunks.append(" ".join(buf))
            buf = [s]
        else:
            buf = trial
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def split_document(text: str, max_sentences: int = 3, max_chars: int = 1200,
                   merge_short: bool = False, short_words: int = 4) -> tuple:
    """-> (parts, chunks).

    parts  : the document split on newline runs, kept verbatim, so the exact
             line-break structure can be reproduced at the end
    chunks : [{cid, part, seq, src, leads}] in reading order
    """
    parts = re.split(r"(\n+)", text)
    chunks, pending = [], []
    for pi, part in enumerate(parts):
        if pi % 2 == 1 or not part.strip():
            continue
        body = part.strip()
        if merge_short and len(body.split()) <= short_words:
            pending.append({"part": pi, "text": body})
            continue
        for seq, piece in enumerate(
                chunk_paragraph(body, max_sentences, max_chars) or [body]):
            leads = pending if seq == 0 else []
            src = "\n".join([l["text"] for l in leads] + [piece])
            chunks.append({"cid": len(chunks), "part": pi, "seq": seq,
                           "src": src, "leads": leads})
            if seq == 0:
                pending = []
    for lead in pending:      # short paragraphs with nothing after them
        chunks.append({"cid": len(chunks), "part": lead["part"], "seq": 0,
                       "src": lead["text"], "leads": []})
    return parts, chunks


def split_lead_output(text: str, leads: list) -> list:
    """Split a translated chunk back into len(leads) + 1 pieces.

    Three deterministic tiers: the model's own line breaks, then a sentence
    boundary (accepted only if the first sentence is about as long as the
    heading it should correspond to), then a cut on the heading's word count so
    reconstruction can never leave a paragraph empty.
    """
    n = len(leads)
    if n <= 0:
        return [text]
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    if len(lines) == n + 1:
        return lines
    sents = split_sentences(text)
    if len(sents) >= n + 1 and all(len(words(s)) <= len(words(l)) + 2
                                   for s, l in zip(sents[:n], leads)):
        return sents[:n] + [" ".join(sents[n:])]
    toks = (text or "").split()
    out, i = [], 0
    for l in leads:
        k = max(1, min(len(words(l)), max(0, len(toks) - i - 1)))
        out.append(" ".join(toks[i:i + k]))
        i += k
    out.append(" ".join(toks[i:]))
    return out


def rebuild(parts: list, chunks: list, finals: dict) -> str:
    """Reconstruct the document, preserving newlines exactly."""
    by_part = {}
    for c in chunks:
        text = (finals.get(c["cid"]) or "").strip()
        pieces = split_lead_output(text, [l["text"] for l in c["leads"]])
        for lead, piece in zip(c["leads"], pieces[:-1]):
            by_part.setdefault(lead["part"], []).append((0, piece))
        by_part.setdefault(c["part"], []).append((c["seq"], pieces[-1]))

    out = []
    for pi, part in enumerate(parts):
        if pi % 2 == 1 or not part.strip():
            out.append(part)
            continue
        body = " ".join(t.strip() for _, t in sorted(by_part.get(pi, []),
                                                     key=lambda x: x[0]) if t.strip())
        lead = part[: len(part) - len(part.lstrip())]
        trail = part[len(part.rstrip()):]
        out.append(lead + (body or part.strip()) + trail)
    return "".join(out)


def newline_signature(text: str) -> list:
    """The exact sequence of newline runs, e.g. ['\\n\\n', '\\n']."""
    return re.findall(r"\n+", text)


def enforce_structure(source: str, translation: str, verbose: int = 1) -> str:
    """Reproduce the source's line-break structure exactly. Paragraph structure
    is a submission requirement, so this repairs rather than raises."""
    if newline_signature(source) == newline_signature(translation):
        return translation
    if verbose:
        print(f"    *** structure repair: source runs {newline_signature(source)} "
              f"!= output runs {newline_signature(translation)}")
    src_parts = re.split(r"(\n+)", source)
    tr_paras = [p for p in re.split(r"\n+", translation) if p.strip()]
    out, k = [], 0
    for i, part in enumerate(src_parts):
        if i % 2 == 1 or not part.strip():
            out.append(part)
        else:
            lead = part[: len(part) - len(part.lstrip())]
            trail = part[len(part.rstrip()):]
            body = tr_paras[k] if k < len(tr_paras) else part.strip()
            out.append(lead + body.strip() + trail)
            k += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Output hygiene
# ---------------------------------------------------------------------------

FMT_OPEN, FMT_CLOSE = "###FORMAT###", "###END_FORMAT###"

_PREAMBLE_RE = re.compile(
    r"^\s*(here is|here's|hona hemen|itzulpena|traducci[oó]n|translation|basque|"
    r"euskara|correcci[oó]n|texto corregido)\b[^\n:]*:\s*", re.IGNORECASE)

_LEAK_MARKERS = (
    "texto original (castellano)", "borrador de traduccion", "borrador de traducción",
    "terminologia aprobada", "terminología aprobada", "corrige el borrador",
    "eres un traductor", "eres un editor", "responde únicamente",
    "here is the translation", "hona hemen itzulpena", FMT_OPEN.lower(),
)


def clean_output(answer: str, keep_lines: bool = False) -> str:
    """Tolerant extractor for models trained to emit a BARE translation."""
    text = (answer or "").strip()
    for tag in (FMT_OPEN, FMT_CLOSE):
        text = text.replace(tag, " ")
    text = _PREAMBLE_RE.sub("", text.strip())
    if keep_lines:
        text = "\n".join(l.strip() for l in text.split("\n") if l.strip())
    else:
        text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip().strip('"').strip()


def looks_like_leak(candidate: str, source: str) -> bool:
    """Prompt scaffolding echoed back, the source returned verbatim, or
    degenerate repetition. The source-echo test is skipped for very short
    segments, where a heading shares its whole vocabulary with the source by
    construction."""
    if not candidate.strip():
        return True
    low = candidate.lower()
    if any(m in low for m in _LEAK_MARKERS):
        return True
    if len(candidate) > len(source) * 2.5 + 60:
        return True
    if len(words(source)) >= 5 and jaccard(candidate, source) > 0.5:
        return True
    toks = candidate.split()
    if len(toks) > 12 and len({w.lower() for w in toks}) <= max(3, len(toks) // 8):
        return True
    return False


def source_carryover(source: str, candidate: str, draft: str = "",
                     n: int = 4) -> Optional[str]:
    """The longest run of SOURCE words reproduced verbatim in an edit but absent
    from the draft it replaces. Catches untranslated Spanish spliced into a
    Basque sentence, which the general leak guard misses because the overlap
    with the whole source stays under its threshold."""
    src, cand = words(source), words(candidate)
    if len(src) < n or len(cand) < n:
        return None
    cand_str = " " + " ".join(cand) + " "
    draft_str = " " + " ".join(words(draft)) + " "
    for i in range(len(src) - n + 1):
        gram = " ".join(src[i:i + n])
        if f" {gram} " in cand_str and f" {gram} " not in draft_str:
            return gram
    return None


def untranslated(source: str, candidate: str) -> bool:
    """True when the output adds nothing to the source. Spanish and Basque share
    almost no vocabulary, so an output whose words are a subset of the source's
    is a copy, not a translation."""
    if not (candidate or "").strip():
        return True
    a, b = set(words(source)), set(words(candidate))
    return bool(a) and b <= a


_SYMBOL_RE = re.compile(r"[0-9⟶→←+=×·%]")


def is_symbolic(text: str) -> bool:
    """A chemical equation or bare formula line: at most one real word, and
    symbols in it. Nothing to translate, and the translator mangles them."""
    ws = re.findall(r"[^\W\d_]{3,}", text or "")
    return len(ws) <= 1 and bool(_SYMBOL_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^<>]{1,40}>|\{[^{}]{1,40}\}|\[[^\[\]]{1,40}\]")


def format_signature(text: str) -> dict:
    t = (text or "").strip()
    return {
        "initial_upper": bool(t[:1].isupper()) if t[:1].isalpha() else None,
        "final_punct": t[-1] if t and not t[-1].isalnum() else "",
        "opens": t[:1] if t[:1] in "\"'«¿¡([" else "",
        "tags": sorted(TAG_RE.findall(t)),
        "brackets": {c: t.count(c) for c in "()[]{}«»\"'"},
        "spaced_units": len(re.findall(r"\d\s+[°%º]", t)),
    }


def format_diff(source: str, translation: str) -> list:
    a, b = format_signature(source), format_signature(translation)
    diffs = []
    if a["initial_upper"] is not None and b["initial_upper"] is not None \
            and a["initial_upper"] != b["initial_upper"]:
        diffs.append("initial capitalisation")
    if a["final_punct"] != b["final_punct"]:
        diffs.append(f"terminal punctuation ({a['final_punct'] or 'none'} in the "
                     f"source, {b['final_punct'] or 'none'} in the translation)")
    if a["opens"] != b["opens"]:
        diffs.append("opening punctuation")
    if a["tags"] != b["tags"]:
        diffs.append(f"tags/placeholders: {a['tags']} vs {b['tags']}")
    for c in "()[]{}«»":
        if a["brackets"][c] != b["brackets"][c]:
            diffs.append(f"unbalanced '{c}'")
            break
    if b["spaced_units"] > a["spaced_units"]:
        diffs.append("space inserted before a degree or percent sign")
    return diffs


def repair_format(source: str, translation: str) -> str:
    """Deterministic, content-free repairs: first-letter case, terminal
    punctuation, stray wrapping quotes, whitespace, and the space that creeps in
    before a degree or percent sign."""
    t = (translation or "").strip()
    if not t:
        return t
    s = source.strip()
    if s[:1] not in "\"'«" and t[:1] in "\"'«" and t[-1:] in "\"'»":
        t = t[1:-1].strip()
    t = re.sub(r"\s+([,;:.!?%°º])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    if s[:1].isalpha() and t[:1].isalpha():
        t = (t[0].upper() + t[1:]) if s[0].isupper() else (t[0].lower() + t[1:])
    src_final = s[-1] if s and not s[-1].isalnum() else ""
    tgt_final = t[-1] if t and not t[-1].isalnum() else ""
    if src_final in ".:;!?" and src_final and src_final != tgt_final:
        t = (t[:-1] if tgt_final and tgt_final in ".:;!?" else t) + src_final
    elif not src_final and tgt_final and tgt_final in ".;":
        t = t[:-1]
    return t.strip()


def same_words(a: str, b: str) -> bool:
    """Word sequence identical up to case and punctuation."""
    return words(a) == words(b)


def match_case(source: str, text: str) -> str:
    if source[:1].isupper() and text[:1].isalpha():
        return text[0].upper() + text[1:]
    return text


def absolutive(term: str) -> str:
    """Basque absolutive singular: stem + -a, with a+a coalescing."""
    t = (term or "").strip()
    return t if t.endswith("a") else t + "a"
