"""Step 1 of the pipeline: deterministic terminology matching.

This is arithmetic, not a model. A glossary entry counts as present in a chunk
when every one of its CONTENT tokens occurs, IN ORDER, with only function words
between them, and each token differs from the glossary form by at most a Spanish
plural or gender ending.

The order requirement and the closed set of permitted endings are what make it
precise. Prefix scoring confirmed "reductor" from *reduciendo*, "transmisión"
from *transmitiendo* and "desplazable" from *desplazados*; none of those is a
plural or a gender variant, so none matches here.

The matcher is deliberately HIGH RECALL at the span level and precise at the
token level: it will still over-generate where a term's words happen to co-occur
in a different noun phrase. Pruning that is the terminology agent's job
(agents.py), which is why this stage is cheap and never calls a model.

No model dependencies: importable and testable without torch.
"""

from __future__ import annotations

import re
from typing import Optional

from .textproc import norm

# ---------------------------------------------------------------------------
# Spanish side
# ---------------------------------------------------------------------------

ES_FUNCTION = {"de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
               "y", "e", "o", "u", "en", "a", "al", "con", "por", "para", "que",
               "se", "su", "sus", "lo", "como", "este", "esta", "estos", "estas",
               "cada", "otro", "otra", "mismo", "misma"}


def tokenise(text: str) -> list:
    return re.findall(r"\w+", norm(text))


def nominal_key(tok: str, fold_gender: bool = True) -> str:
    """Fold a Spanish nominal token onto a form shared by its inflections.

        motores -> motor      luces   -> luz       frenos  -> fren~
        altura  -> altur~     alturas -> altur~    aceites -> aceite

    Plural first, then gender. The trailing tilde marks a folded gender vowel so
    that 'freno' and 'frena' meet while 'freno' and 'fren' do not.
    """
    t = tok
    # -ces -> -z  (luces -> luz)
    if len(t) > 4 and t.endswith("ces"):
        t = t[:-3] + "z"
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    # A trailing -e is then folded away, which is what lets the two Spanish
    # plural patterns meet on one key without having to know which noun takes
    # which:  motores -> motore -> motor  and  motor -> motor
    #         aceites -> aceite -> aceit  and  aceite -> aceit
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    if fold_gender and len(t) > 4 and t[-1] in "oa":
        t = t[:-1] + "~"
    return t


def content_keys(text: str, fold_gender: bool = True) -> list:
    return [nominal_key(t, fold_gender) for t in tokenise(text)
            if t not in ES_FUNCTION]


def find_term(term_keys: list, text_tokens: list, text_keys: list,
              max_gap: int = 0) -> Optional[tuple]:
    """Sequence match with function-word skipping -> (start, end) or None."""
    if not term_keys:
        return None
    n = len(text_keys)
    for start in range(n):
        if text_keys[start] != term_keys[0]:
            continue
        pos, ok = start, True
        for key in term_keys[1:]:
            nxt, gap, found = pos + 1, 0, False
            while nxt < n:
                if text_keys[nxt] == key:
                    found = True
                    break
                if text_tokens[nxt] not in ES_FUNCTION:
                    gap += 1          # a content word costs, a function word is free
                    if gap > max_gap:
                        break
                nxt += 1
            if not found:
                ok = False
                break
            pos = nxt
        if ok:
            return (start, pos + 1)
    return None


# ---------------------------------------------------------------------------
# Basque side
# ---------------------------------------------------------------------------

EU_SUFFIXES = sorted(
    ["arentzat", "entzat", "arekin", "etatik", "etara", "ekin", "aren", "ari",
     "etan", "etik", "ean", "era", "tik", "tan", "ren", "ak", "ei", "ek", "en",
     "ez", "ra", "an", "ko", "a", "k", "n", "z"], key=len, reverse=True)
EU_TAILS = set(EU_SUFFIXES) | {""}


def _target_regexes(target: str):
    """Basque inflects by suffixation, so the citation form is a prefix of every
    declined form: 'kalandra' -> 'kalandrak', 'kalandraren', 'kalandratik'.
    Matching the final token with a trailing \\w* recognises the paradigm without
    generating it. Compounds appear with or without a hyphen."""
    toks = tokenise(target)
    if not toks:
        return None, None, ""
    sep = r"[\s\-]+"
    body = sep.join(re.escape(t) for t in toks[:-1])
    tail = re.escape(toks[-1]) + r"\w*"
    full = re.compile(r"(?<!\w)" + (body + sep if body else "") + tail)
    head = re.compile(r"(?<!\w)" + tail) if len(toks) > 1 else full
    return full, head, toks[-1]


def _plausible(observed: str, last_token: str) -> bool:
    """Guard for short targets: 'auto' would otherwise match *Automotive* in an
    English society name."""
    if len(last_token) > 5:
        return True
    tail = norm(observed).split()[-1][len(last_token):]
    if tail in EU_TAILS:
        return True
    # Basque doubles a final consonant before a vowel-initial case ending:
    # motor + -ak -> motorrak. Without this, every declined form of a short
    # consonant-final term is scored absent.
    if tail and last_token and tail[0] == last_token[-1] and tail[1:] in EU_TAILS:
        return True
    return False


def match_target(target: str, norm_text: str) -> dict:
    """-> status "present" (whole term, any inflection), "partial" (head noun
    only) or "absent"."""
    full, head, last = _target_regexes(target)
    if full is None:
        return {"status": "absent", "observed": []}
    for rx, status in ((full, "present"), (head, "partial")):
        if rx is None:
            continue
        for m in rx.finditer(norm_text):
            if _plausible(m.group(0), last):
                return {"status": status, "observed": [m.group(0)]}
    return {"status": "absent", "observed": []}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class TerminologyDatabase:
    """Read-only index over ONE dictionary ({src: [approved targets]}), built
    once per run and handing out copies so nothing leaks between documents."""

    def __init__(self, dictionary: dict, max_gap: int = 0):
        self.dict = {s: list(t) for s, t in dictionary.items()}
        self.max_gap = max_gap
        self.sources = list(self.dict.keys())
        self._strict = {s: content_keys(s, False) for s in self.sources}
        self._folded = {s: content_keys(s, True) for s in self.sources}
        # Entries whose content tokens are all function words would match
        # everywhere; drop them.
        self._usable = [s for s in self.sources if self._strict[s]]
        # Which entry, if any, owns a strict key sequence outright. Stops
        # "anilla" claiming *anillos* on a gender fold when "anillo" is itself
        # in the glossary.
        self._owner = {}
        for s in self._usable:
            self._owner.setdefault(tuple(self._strict[s]), s)
        # First content key -> entries, for cheap near-miss lookup.
        self._by_first = {}
        for s in self._usable:
            for k in set(self._strict[s]):
                self._by_first.setdefault(k, []).append(s)

    # --- matching ---------------------------------------------------------

    def _find(self, term, tokens, strict, folded):
        span = find_term(self._strict[term], tokens, strict, self.max_gap)
        if span is not None:
            return span
        span = find_term(self._folded[term], tokens, folded, self.max_gap)
        if span is None:
            return None
        content = tuple(strict[i] for i in range(*span)
                        if tokens[i] not in ES_FUNCTION)
        owner = self._owner.get(content)
        return None if (owner is not None and owner != term) else span

    def _hits(self, text: str) -> list:
        tokens = tokenise(text)
        strict = [nominal_key(t, False) for t in tokens]
        folded = [nominal_key(t, True) for t in tokens]
        out = []
        for s in self._usable:
            span = self._find(s, tokens, strict, folded)
            if span is not None:
                out.append((len(self._strict[s]), span, s, tokens))
        out.sort(key=lambda h: (-h[0], h[1][0]))
        return out

    def match(self, text: str) -> dict:
        """-> {source term: [approved targets]} for the entries present in this
        chunk, longest first."""
        return {s: list(self.dict[s]) for _, _, s, _ in self._hits(text)}

    def match_spans(self, text: str) -> dict:
        """Same, but returning the matched surface span, for inspection."""
        return {s: " ".join(toks[span[0]:span[1]])
                for _, span, s, toks in self._hits(text)}

    # --- recall support for the terminology agent -------------------------

    def near_misses(self, text: str, exclude: dict, limit: int = 10) -> dict:
        """Glossary entries NOT matched in this chunk but sharing at least one
        content key with it, ranked by how much of the entry is present.

        These are offered to the terminology agent as recovery candidates: the
        matcher requires every token in order, so a coordinated or reordered
        mention ("sistemas de frenado y de refrigeración") is missed, and only a
        model reading the sentence can decide whether the term is really there.
        """
        keys = set(content_keys(text, False)) | set(content_keys(text, True))
        if not keys:
            return {}
        scored = []
        seen = set(exclude or {})
        for k in keys:
            for s in self._by_first.get(k, []):
                if s in seen:
                    continue
                seen.add(s)
                need = self._strict[s]
                if not need:
                    continue
                overlap = sum(1 for t in need if t in keys) / len(need)
                if overlap >= 0.5:
                    scored.append((overlap, len(need), s))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return {s: list(self.dict[s]) for _, _, s in scored[:limit]}
