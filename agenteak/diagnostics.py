"""Report-only diagnostics.

These check nothing and block nothing. The editor is a fine-tuned model doing
the job it was trained for and its output ships as it stands; what follows only
measures the result, so a run can be inspected afterwards without reading every
segment by hand. Everything here lands in the stats sidecar.

Note: this is NOT the terminology success rate scorer used in the paper. That is
the official WMT25 scorer with our lemma-matching modifications, and it is run
separately on the finished submission. What follows is a cheap surface check
intended to flag segments worth looking at.

No model dependencies.
"""

from __future__ import annotations

import re
from collections import Counter

from .terminology import match_target
from .textproc import norm, number_mismatch, words

# ---------------------------------------------------------------------------


def coverage_check(source: str, translation: str) -> dict:
    """Target/source content-word ratio. Basque runs shorter than Spanish, so
    the floor sits below parity: this notices collapse, not compression."""
    src_words = len([w for w in re.findall(r"\w+", source) if len(w) > 1])
    tgt_words = len([w for w in re.findall(r"\w+", translation) if len(w) > 1])
    if src_words == 0:
        return {"ok": True, "ratio": 1.0, "src_words": 0, "tgt_words": tgt_words}
    ratio = tgt_words / src_words
    return {"ok": ratio >= 0.60, "ratio": round(ratio, 2),
            "src_words": src_words, "tgt_words": tgt_words}


# Basque auxiliaries and light verbs that legitimately end a sentence.
EU_FINAL_OK = {"da", "du", "dira", "ditu", "dute", "zen", "ziren", "zuen", "zuten",
               "dago", "daude", "izan", "egin", "behar", "ahal", "dezake",
               "daiteke", "dio", "diote", "zaio", "zaie", "ari", "gabe", "arte",
               "ere"}


def repetition_check(translation: str) -> dict:
    """Garbled output where a phrase is duplicated in place of another.
    Target-internal by necessity: Spanish and Basque share no vocabulary, so a
    word count in one says nothing about the other."""
    ws = words(translation)
    content = [w for w in ws if len(w) >= 5]
    if len(ws) < 5:
        return {"ok": True, "repeated": []}
    stems = Counter(w[:7] for w in content)
    tripled = [st for st, n in stems.items() if n >= 3]
    if tripled:
        return {"ok": False, "repeated": sorted(tripled)[:4]}
    if content and ws[-1] not in EU_FINAL_OK:
        first, last = content[0], ws[-1]
        if first == last or (len(first) >= 7 and len(last) >= 6
                             and first[:6] == last[:6] and first != last):
            return {"ok": False, "repeated": sorted({first, last})}
    dup = [f"{a} {b}" for (a, b), n in Counter(zip(content, content[1:])).items()
           if n >= 2]
    if dup:
        return {"ok": False, "repeated": sorted(dup)[:3]}
    return {"ok": True, "repeated": []}


CASE_SUFFIXES = sorted(
    ["arentzat", "entzat", "arekin", "etatik", "etara", "ekin", "aren", "ari",
     "etan", "etik", "ean", "era", "tik", "tan", "ren", "ak", "ei", "ek", "en",
     "ez", "ra", "an", "ko", "a", "k", "n", "z"], key=len, reverse=True)


def stem_of(form: str) -> str:
    """Strip one case ending to expose the stem the writer actually used.
    Truncating to a fixed prefix does not work: 'karteraren' and 'karterretik'
    agree on six characters and differ exactly where it matters."""
    f = norm(form).split()[-1] if (form or "").strip() else ""
    for suf in CASE_SUFFIXES:
        if f.endswith(suf) and len(f) - len(suf) >= 3:
            return f[: -len(suf)]
    return f


def diagnose_chunk(source: str, translation: str, terms: dict) -> dict:
    """-> {"issues": [...], "forms": {es -> observed Basque form}} for ONE chunk,
    judged against the terminology recorded for that same chunk."""
    issues, forms = [], {}

    nm = number_mismatch(source, translation)
    if not nm["ok"]:
        issues.append({"type": "numbers",
                       "detail": f"missing {nm['missing']}, added {nm['added']}"})

    cov = coverage_check(source, translation)
    if not cov["ok"]:
        issues.append({"type": "omission",
                       "detail": f"{cov['tgt_words']} target words for "
                                 f"{cov['src_words']} source words "
                                 f"(ratio {cov['ratio']})"})

    rep = repetition_check(translation)
    if not rep["ok"]:
        issues.append({"type": "repetition", "detail": ", ".join(rep["repeated"])})

    norm_t = norm(translation)
    rank = {"present": 2, "partial": 1, "absent": 0}
    for src_term, targets in (terms or {}).items():
        targets = targets if isinstance(targets, list) else [targets]
        best = {"status": "absent", "observed": [],
                "target": targets[0] if targets else ""}
        for tgt_term in targets:
            r = match_target(tgt_term, norm_t)
            if rank[r["status"]] > rank[best["status"]]:
                best = {**r, "target": tgt_term}
            if r["status"] == "present":
                break
        if best["observed"]:
            forms[src_term] = best["observed"][0]
        if best["status"] == "absent":
            issues.append({"type": "term_missing",
                           "detail": f'"{src_term}" -> {best["target"]} does not appear',
                           "term": src_term})
        elif best["status"] == "partial":
            issues.append({"type": "term_form",
                           "detail": f'"{src_term}" appears only as {best["observed"][0]}',
                           "term": src_term})
    return {"issues": issues, "forms": forms}


def consistency_report(forms_by_chunk: dict) -> dict:
    """Same term, same stem across one document. Case variation is expected;
    stem variation ('karteraren' beside 'karterretik') is the thing worth
    seeing, and it is only visible once the whole document is finished."""
    by_term = {}
    for forms in forms_by_chunk.values():
        for src_term, form in (forms or {}).items():
            stem = stem_of(form)
            if stem:
                by_term.setdefault(src_term, Counter())[stem] += 1
    inconsistent = {t: sorted(c) for t, c in by_term.items() if len(c) > 1}
    return {"ok": not inconsistent, "inconsistent": inconsistent}
