"""Prompts for the three agents.

ONE OF THESE IS FROZEN. The translator prompt is reproduced CHARACTER FOR
CHARACTER from the fine-tuning data; it is not a style choice and must not be
tidied, because the conditioning the fine-tune bought fires on that exact
string.

The editor and terminology agent prompts are ordinary engineering. Both roles
are served by UN-FINE-TUNED models — base Latxa-8B-Instruct and Qwen3-4B — so
their prompts and few-shot examples can be edited, extended, or replaced per
domain. Only the translator carries adapters.

    >>> NOTE ON THE FEW-SHOT EXAMPLES <<<
    The Basque in FEWSHOT_* below is illustrative. It is intended to show the
    agent the SHAPE of each decision (reject / recover / disambiguate), not to
    teach it specific terminology. Before a submission run, have a Basque
    speaker check these, or replace them with pairs drawn from your own
    glossary — a wrong example here is applied silently to every chunk.
"""

from __future__ import annotations

import json

from .config import (DOMAIN_PHRASE, DOMAIN_TAG, EDITOR_DOMAIN_PHRASE,
                     canonical_domain)
from .textproc import FMT_CLOSE, FMT_OPEN

# ===========================================================================
# Agent 2 — translator  (FROZEN: matches the fine-tuning data)
# ===========================================================================


def translator_system_prompt(domain: str) -> str:
    """Exactly the system prompt the translator was trained with."""
    domain = canonical_domain(domain)
    if domain in DOMAIN_PHRASE:
        opening = ("Eres un traductor experto del castellano al euskera (euskara batua), "
                   f"especializado en {DOMAIN_PHRASE[domain]}.")
        terminologia = "y empleando la terminología asentada de ese ámbito. "
    else:
        opening = ("Eres un traductor experto del castellano al euskera (euskara batua), "
                   "capaz de trabajar con textos de cualquier ámbito.")
        terminologia = "y empleando un registro estándar. "
    return (
        opening + "\n"
        f"Ámbito: {DOMAIN_TAG.get(domain, domain)}.\n"
        "Traduce con precisión y naturalidad, respetando la declinación y la sintaxis propias "
        "del euskera, " + terminologia +
        "Conserva exactamente todas las cifras, unidades y nombres propios tal como aparecen "
        "en el original, así como el formato del texto de partida (mayúsculas, puntuación, "
        "saltos de línea y etiquetas). "
        "No añadas explicaciones, comentarios ni el texto de partida. "
        "Responde únicamente con la traducción."
    )


# ===========================================================================
# Agent 3 — editor / verifier  (NOT frozen: base Latxa-8B-Instruct, zero-shot)
# ===========================================================================


def editor_system_prompt(domain: str, with_terms: bool) -> str:
    """System prompt for the editor.

    The wording is inherited from an earlier fine-tuned editor that was not used
    in the end, which is why the scaffolding is unaccented (`declinacion`,
    `anadidos`). That is a harmless inheritance, not a requirement: the editor is
    the base instruct model, so this prompt may be edited freely.

    The no-terminology variant drops clause (1) and the approved-term mention
    and changes nothing else: in noterm mode the pipeline must not carry any
    terminology knowledge at all, not even the shape of a term list.
    """
    phrase = EDITOR_DOMAIN_PHRASE.get(canonical_domain(domain))
    opening = ("Eres un editor experto de traducciones del castellano al euskera"
               + (f", especializado en {phrase}. " if phrase else ". "))
    if with_terms:
        return (opening
                + "Se te proporciona el texto original en castellano, un borrador de "
                  "traduccion al euskera y una lista de terminologia aprobada (castellano -> "
                  "euskera). Tu tarea es corregir el borrador para que: (1) use exclusivamente "
                  "la terminologia aprobada indicada, (2) tenga la declinacion "
                  "(kasu-atzizkiak) correcta en euskera, y (3) no contenga omisiones ni "
                  "anadidos respecto al texto original. Si el borrador ya es correcto, "
                  "devuelvelo tal cual, sin introducir cambios innecesarios. Devuelve "
                  "unicamente la traduccion al euskera corregida, sin explicaciones.")
    return (opening
            + "Se te proporciona el texto original en castellano y un borrador de traduccion "
              "al euskera. Tu tarea es corregir el borrador para que: (1) tenga la declinacion "
              "(kasu-atzizkiak) correcta en euskera, y (2) no contenga omisiones ni anadidos "
              "respecto al texto original. Si el borrador ya es correcto, devuelvelo tal cual, "
              "sin introducir cambios innecesarios. Devuelve unicamente la traduccion al "
              "euskera corregida, sin explicaciones.")


def editor_user_prompt(source: str, draft: str, present: dict) -> str:
    """Exactly the user-turn shape of the editor's training data."""
    if present:
        terms = "\n".join(f"- {src} -> {' / '.join(targets)}"
                          for src, targets in present.items())
        return (f"Texto original (castellano):\n{source}\n\n"
                f"Borrador de traduccion (euskera):\n{draft}\n\n"
                f"Terminologia aprobada:\n{terms}\n\n"
                "Corrige el borrador aplicando la terminologia aprobada y la declinacion "
                "correcta, y eliminando cualquier omision o anadido respecto al texto "
                "original.")
    return (f"Texto original (castellano):\n{source}\n\n"
            f"Borrador de traduccion (euskera):\n{draft}\n\n"
            "Corrige el borrador aplicando la declinacion correcta y eliminando cualquier "
            "omision o anadido respecto al texto original.")


# ---------------------------------------------------------------------------
# Editor few-shot examples
# ---------------------------------------------------------------------------
# The editor is an un-fine-tuned model, so these carry real weight and are used
# by default. Each example is (user_turn, assistant_turn), and the three cover
# the failure modes observed on this corpus: a term present but in citation form
# where the syntax needs a case ending; a term absent from the draft entirely;
# and a draft that is already correct and must be returned untouched — the last
# matters most, since an idle editor that rewrites anyway is the expensive
# failure.
#
# They are skipped in noterm mode, where a prompt showing approved-term blocks
# would be off-distribution for the task actually being asked.
#
# The Basque here is illustrative; see the note at the top of this file.

FEWSHOT_EDITOR = [
    (
        "Texto original (castellano):\n"
        "El motor de combustión interna transforma la energía química en energía mecánica.\n\n"
        "Borrador de traduccion (euskera):\n"
        "Barne-errekuntzako motor energia kimikoa energia mekaniko bihurtzen du.\n\n"
        "Terminologia aprobada:\n"
        "- motor de combustión interna -> barne-errekuntzako motor\n\n"
        "Corrige el borrador aplicando la terminologia aprobada y la declinacion "
        "correcta, y eliminando cualquier omision o anadido respecto al texto original.",
        "Barne-errekuntzako motorrak energia kimikoa energia mekaniko bihurtzen du."
    ),
    (
        "Texto original (castellano):\n"
        "La batería alimenta el sistema de arranque del vehículo.\n\n"
        "Borrador de traduccion (euskera):\n"
        "Pilak ibilgailuaren abiarazte-sistema elikatzen du.\n\n"
        "Terminologia aprobada:\n"
        "- batería -> bateria\n\n"
        "Corrige el borrador aplicando la terminologia aprobada y la declinacion "
        "correcta, y eliminando cualquier omision o anadido respecto al texto original.",
        "Bateriak ibilgailuaren abiarazte-sistema elikatzen du."
    ),
    (
        "Texto original (castellano):\n"
        "Las energías renovables representan una parte creciente de la producción.\n\n"
        "Borrador de traduccion (euskera):\n"
        "Energia berriztagarriek ekoizpenaren zati gero eta handiagoa dira.\n\n"
        "Terminologia aprobada:\n"
        "- energía renovable -> energia berriztagarri\n\n"
        "Corrige el borrador aplicando la terminologia aprobada y la declinacion "
        "correcta, y eliminando cualquier omision o anadido respecto al texto original.",
        "Energia berriztagarriek ekoizpenaren zati gero eta handiagoa dira."
    ),
]


def editor_messages(domain: str, with_terms: bool, source: str, draft: str,
                    present: dict, few_shot: bool = True) -> list:
    """Full message list for the editor."""
    msgs = [{"role": "system", "content": editor_system_prompt(domain, with_terms)}]
    if few_shot and with_terms:
        for user, assistant in FEWSHOT_EDITOR:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
    msgs.append({"role": "user",
                 "content": editor_user_prompt(source, draft, present)})
    return msgs


# ===========================================================================
# Agent 1 — terminology agent  (NOT frozen: zero-shot model, editable)
# ===========================================================================
# Step 1 hands this agent a high-recall candidate list. The agent has three
# jobs, and the few-shot examples below show one of each:
#
#   REJECT       a candidate whose words co-occur in the chunk but do not form
#                the term ("aceite movido por el motor" is not "aceite de motor")
#   RECOVER      a term the matcher missed because the mention is coordinated or
#                reordered ("sistemas de frenado y de refrigeración")
#   DISAMBIGUATE choose ONE target where the glossary offers several
#
# Output is JSON so it can be parsed deterministically. A model that returns
# anything unparseable falls back to the matcher's output, so a bad response
# degrades the run to the deterministic baseline rather than breaking it.

TERM_AGENT_SYSTEM = (
    "You are a terminology specialist for Spanish-to-Basque technical translation.\n\n"
    "You are given a Spanish source passage, a list of CANDIDATE glossary entries that a "
    "string matcher found in it, and a list of POSSIBLE entries the matcher may have missed. "
    "Each entry has one or more approved Basque targets.\n\n"
    "Your job, for this passage only:\n"
    "1. REJECT candidates that are not genuine uses of the term. The matcher works on word "
    "sequences, so it can fire when the words of a term happen to co-occur inside a different "
    "noun phrase.\n"
    "2. RECOVER possible entries that ARE genuinely used in the passage, even if the wording "
    "differs (coordination, reordering, a synonym of a modifier).\n"
    "3. DISAMBIGUATE: where an entry lists several approved Basque targets, choose exactly one, "
    "the one that fits this context.\n\n"
    "Judge only what this passage says. Do not translate the passage. Do not invent glossary "
    "entries: every term you return must come from the two lists you were given.\n\n"
    "Respond with JSON only, no commentary, in exactly this form:\n"
    '{"terms": [{"es": "<source term, copied exactly>", "eu": "<one chosen target>"}]}\n'
    'If no entry applies, respond with {"terms": []}.'
)


def _fewshot_user(source: str, candidates: dict, possible: dict) -> str:
    return term_agent_user_prompt(source, candidates, possible)


def _fewshot_assistant(pairs: list) -> str:
    return json.dumps({"terms": [{"es": a, "eu": b} for a, b in pairs]},
                      ensure_ascii=False)


# Each entry: (source, candidates, possible, chosen_pairs)
_FEWSHOT_TERM_SPEC = [
    # 1. REJECT — the matcher fired on a different noun phrase.
    (
        "La bomba de aceite movida por el motor mantiene la presión del circuito.",
        {"aceite de motor": ["motor-olio"], "bomba de aceite": ["olio-ponpa"]},
        {},
        [("bomba de aceite", "olio-ponpa")],
    ),
    # 2. RECOVER — coordination split the term, so the matcher missed it.
    (
        "El vehículo incorpora sistemas de frenado y de refrigeración independientes.",
        {},
        {"sistema de refrigeración": ["hozte-sistema"],
         "sistema de frenado": ["balaztatze-sistema"]},
        [("sistema de frenado", "balaztatze-sistema"),
         ("sistema de refrigeración", "hozte-sistema")],
    ),
    # 3. DISAMBIGUATE — the entry offers two targets; context selects one.
    (
        "La red eléctrica de alta tensión conecta la central con las subestaciones.",
        {"red eléctrica": ["sare elektriko", "elektrizitate-sare"],
         "alta tensión": ["goi-tentsio"]},
        {},
        [("red eléctrica", "sare elektriko"), ("alta tensión", "goi-tentsio")],
    ),
]


def term_agent_user_prompt(source: str, candidates: dict, possible: dict) -> str:
    """One chunk, its candidate entries and its near-miss entries."""
    def _block(d):
        if not d:
            return "  (none)"
        return "\n".join(f"  - {src} -> {' / '.join(targets)}"
                         for src, targets in d.items())
    return (f"PASSAGE (Spanish):\n{source}\n\n"
            f"CANDIDATE entries found by the matcher:\n{_block(candidates)}\n\n"
            f"POSSIBLE entries the matcher may have missed:\n{_block(possible)}\n\n"
            "Return the entries genuinely used in this passage, one chosen Basque target "
            "each, as JSON.")


def term_agent_messages(source: str, candidates: dict, possible: dict,
                        few_shot: bool = True) -> list:
    """Full message list for the terminology agent."""
    msgs = [{"role": "system", "content": TERM_AGENT_SYSTEM}]
    if few_shot:
        for src, cand, poss, chosen in _FEWSHOT_TERM_SPEC:
            msgs.append({"role": "user", "content": _fewshot_user(src, cand, poss)})
            msgs.append({"role": "assistant", "content": _fewshot_assistant(chosen)})
    msgs.append({"role": "user",
                 "content": term_agent_user_prompt(source, candidates, possible)})
    return msgs


# ===========================================================================
# Optional formatting pass
# ===========================================================================

FMT_SYSTEM = (
    "You are checking the FORMATTING of a Basque translation against its Spanish source. "
    "You are not a translator and not an editor: the words are already correct.\n\n"
    "You may change ONLY: capitalisation, punctuation, quotation marks, brackets, spacing, "
    "and the position of tags or placeholders.\n"
    "You may NOT: add, remove, reorder, translate or replace a single word.\n\n"
    "If the translation already matches the source's formatting, return it unchanged.\n\n"
    f"Respond ONLY with the text between {FMT_OPEN} and {FMT_CLOSE}:\n"
    f"{FMT_OPEN}\n<Basque text with corrected formatting>\n{FMT_CLOSE}")
