"""Run configuration for the Agenteak pipeline.

Everything tunable lives here. `Config` is a plain dataclass, so a run can be
configured from the CLI, from a YAML/JSON file, or from Python.

The domain strings and prompt shapes are NOT style choices: they reproduce the
strings the two 8B agents were fine-tuned on, and the domain conditioning only
fires on these exact forms. See prompts.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

TRANSLATOR_MODEL_ID = "jurgiraud/latxa-eseu-wmt26-augmented"
# The editor is the BASE instruct model, prompted zero-shot with few-shot
# examples. Only the translator is fine-tuned.
EDITOR_MODEL_ID = "HiTZ/Latxa-Llama-3.1-8B-Instruct"
BASE_MODEL_ID = "HiTZ/Latxa-Llama-3.1-8B-Instruct"

# Step 2 of the pipeline: the terminology agent. A small reasoning model is
# enough, because the task is selection and disambiguation rather than
# generation.
TERM_AGENT_MODEL_ID = "Qwen/Qwen3-4B"

# If the fine-tunes are also published as LoRA adapters, set these and both 8B
# roles are served from ONE resident base model with an adapter switch.
TRANSLATOR_ADAPTER: Optional[str] = None
EDITOR_ADAPTER: Optional[str] = None

# ---------------------------------------------------------------------------
# Domain strings (must match the fine-tuning data)
# ---------------------------------------------------------------------------

DOMAIN_PHRASE = {
    "automotive": "el ámbito de la automoción",
    "energy": "el ámbito de la energía",
}
DOMAIN_TAG = {
    "automotive": "automoción",
    "energy": "energía",
}
EDITOR_DOMAIN_PHRASE = {          # editor scaffolding is unaccented, as trained
    "automotive": "el ambito de la automocion",
    "energy": "el ambito de la energia",
}

# The shared task uses "automotion"; we accept both spellings everywhere.
DOMAIN_ALIASES = {"automotion": "automotive", "automocion": "automotive",
                  "automoción": "automotive", "energia": "energy",
                  "energía": "energy"}


def canonical_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    return DOMAIN_ALIASES.get(d, d)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
# proper / random / noterm are the three shared-task conditions.
# base_noterm is a contrastive run of our own: the un-fine-tuned base model
# with no terminology anywhere.

MODE_SPECS = {
    "proper":      {"terms": "proper", "weights": "ft"},
    "random":      {"terms": "random", "weights": "ft"},
    "noterm":      {"terms": None,     "weights": "ft"},
    "base_noterm": {"terms": None,     "weights": "base"},
}

# ---------------------------------------------------------------------------
# Curated section headings, Spanish -> Basque
# ---------------------------------------------------------------------------
# Wikipedia-style section titles that recur across documents and are not in the
# terminology glossary. VERIFY THESE AGAINST YOUR OWN BASQUE before a
# submission run: a wrong entry is applied silently and everywhere.

HEADING_GLOSS = {
    "uso": "Erabilera", "usos": "Erabilerak",
    "grado": "Gradua", "grados": "Graduak",
    "historia": "Historia", "introducción": "Sarrera",
    "descripción": "Deskribapena", "definición": "Definizioa",
    "características": "Ezaugarriak", "funcionamiento": "Funtzionamendua",
    "tipos": "Motak", "clasificación": "Sailkapena",
    "aplicaciones": "Aplikazioak", "componentes": "Osagaiak",
    "mantenimiento": "Mantentze-lanak", "ventajas": "Abantailak",
    "desventajas": "Desabantailak", "seguridad": "Segurtasuna",
    "producción": "Ekoizpena", "energía": "Energia",
    "eficiencia": "Eraginkortasuna", "impacto ambiental": "Ingurumen-inpaktua",
    "medio ambiente": "Ingurumena", "referencias": "Erreferentziak",
    "véase también": "Ikus, gainera", "enlaces externos": "Kanpo estekak",
    "bibliografía": "Bibliografia", "notas": "Oharrak", "galería": "Galeria",
}


@dataclass
class Config:
    """One run of the pipeline."""

    # --- what to run ------------------------------------------------------
    domain: str = "automotive"
    direction: str = "eseu"
    modes: tuple = ("proper",)
    seed: int = 42
    limit_docs: Optional[int] = None
    verbose: int = 1                  # 0 per-document, 1 per-stage, 2 + chunks

    # --- paths ------------------------------------------------------------
    text_path: Optional[str] = None   # shared-task JSON, .txt file, or directory
    terms_path: Optional[str] = None  # shared-task JSON, or .json/.tsv glossary
    out_dir: str = "output"
    system_name: str = "agenteak"

    # --- models -----------------------------------------------------------
    translator_model: str = TRANSLATOR_MODEL_ID
    editor_model: str = EDITOR_MODEL_ID
    base_model: str = BASE_MODEL_ID
    term_agent_model: str = TERM_AGENT_MODEL_ID
    translator_adapter: Optional[str] = TRANSLATOR_ADAPTER
    editor_adapter: Optional[str] = EDITOR_ADAPTER

    # --- stage 2: terminology agent ---------------------------------------
    # The deterministic matcher is high-recall by design. The agent prunes its
    # false positives, recovers entries it missed, and picks one target where a
    # glossary entry offers several. Disable it to fall back to the matcher
    # alone (faster, and the pipeline still runs end to end).
    term_agent: bool = True
    term_agent_thinking: bool = True
    # How many unmatched glossary entries to offer the agent as recovery
    # candidates. These are entries sharing at least one content word with the
    # chunk. Higher = better recall, longer prompt.
    term_agent_recall_candidates: int = 10

    # --- stage 5: editor --------------------------------------------------
    # The editor is the base instruct model, not a fine-tune, so few-shot
    # examples do real work here. Disable to compare against a bare prompt, or
    # if you point --editor-model at a fine-tuned editor of your own.
    editor_few_shot: bool = True

    # --- chunking ---------------------------------------------------------
    max_sentences: int = 3
    max_chunk_chars: int = 1200
    merge_short_paragraphs: bool = False
    short_paragraph_words: int = 4
    heading_context_chars: int = 400
    heading_absolutive: bool = True
    verify_short_paragraphs: bool = True

    # --- terminology matching ---------------------------------------------
    # Content words permitted between two tokens of a term. Function words are
    # always free. 0 is precision-first and the right default.
    term_max_gap: int = 0

    # --- behaviour --------------------------------------------------------
    format_repair: bool = True        # deterministic capitalisation/punctuation
    diagnostics: bool = True          # report-only checks -> stats sidecar
    dump_memory: bool = False         # write the per-document store

    # --- generation budgets (set from the hardware profile) ---------------
    load_in_4bit: Optional[bool] = None
    think_budget: Optional[int] = None
    answer_budget: Optional[int] = None
    max_resident_large: Optional[int] = None

    heading_gloss: dict = field(default_factory=lambda: dict(HEADING_GLOSS))

    def __post_init__(self):
        self.domain = canonical_domain(self.domain)
        if isinstance(self.modes, str):
            self.modes = (self.modes,)
        for m in self.modes:
            if m not in MODE_SPECS:
                raise ValueError(f"unknown mode {m!r}; expected one of {list(MODE_SPECS)}")

    def to_dict(self) -> dict:
        return asdict(self)
