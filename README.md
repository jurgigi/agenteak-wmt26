# Agenteak

A multi-agent pipeline for terminology-constrained **Spanish → Basque** translation, built for the [WMT26 Terminology Translation Task](https://www2.statmt.org/wmt26/) (Track 1, es→eu, automotive and energy domains).

Three compact open-weight models cooperate on [LangGraph](https://github.com/langchain-ai/langgraph). None is larger than 8B parameters, so the whole system runs on a single consumer GPU.

```
document
  │
  ├─ prepare     split into chunks of ≤3 sentences
  ├─ match       DETERMINISTIC matcher: which glossary entries occur here?
  ├─ select      TERMINOLOGY AGENT  (Qwen3-4B, reasoning)   prune · recover · disambiguate
  ├─ translate   TRANSLATOR         (Latxa-8B, fine-tuned)  chunk by chunk
  ├─ verify      EDITOR             (Latxa-8B)              terminology + declension repair
  ├─ format      deterministic capitalisation / punctuation repair
  └─ finalize    rebuild the document, run report-only diagnostics
```

Terminology is handled as a **soft** constraint. Basque is agglutinative: a glossary entry such as `barne-errekuntzako motor` surfaces as `motorrak`, `motorraren`, `motorrean`, so copying a target string verbatim produces ungrammatical output. The terminology agent selects terms, the translator is told to inflect them as the sentence requires, and the editor checks they arrived.

---

## Install

```bash
git clone https://github.com/<you>/agenteak-wmt26.git
cd agenteak-wmt26
pip install -r requirements.txt
```

Python 3.9+. A CUDA GPU is strongly recommended — see [Hardware](#hardware).

The models are pulled from the Hugging Face Hub on first run:

| Role | Model | Size |
|---|---|---|
| Terminology agent | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | 4B |
| Translator | [`jurgiraud/latxa-eseu-wmt26-augmented`](https://huggingface.co/jurgiraud/latxa-eseu-wmt26-augmented) | 8B |
| Editor | [`HiTZ/Latxa-Llama-3.1-8B-Instruct`](https://huggingface.co/HiTZ/Latxa-Llama-3.1-8B-Instruct) | 8B |

Latxa is gated under the Llama 3.1 licence, so accept the terms on the Hub and log in first:

```bash
huggingface-cli login
```

---

## Quick start

### Shared-task data

```bash
python -m agenteak \
    --domain automotive \
    --mode proper \
    --input  data/text.automotion.eseu.json \
    --terms  data/terms.automotion.eseu.json \
    --out    output/
```

Writes `output/agenteak.proper.automotive.eseu.json` — a JSON list of translations, one per input document, in the same order and with the same paragraph structure.

### Your own text

```bash
python -m agenteak --domain energy --mode noterm --input my_article.txt
python -m agenteak --domain energy --mode proper --input texts/ --terms glossary.tsv
```

A `.txt` file is one document; a directory is one document per `.txt`, sorted by filename. In text mode you get both the JSON list and a folder of `.txt` files.

### Try it on the examples

```bash
python -m agenteak -d automotive -m proper \
    -i examples/text.automotive.eseu.json \
    -t examples/terms.automotive.eseu.json --limit 1 -v 2
```

`--limit` truncates the corpus for a trial run and tags the filename so a partial run can never be mistaken for a submission.

---

## Input formats

### Shared-task JSON

```jsonc
// text.<domain>.<direction>.json
["First document…", "Second document…"]
```

```jsonc
// terms.<domain>.<direction>.json
{
  "proper": {"motor de combustión interna": ["barne-errekuntzako motor"]},
  "random": {"motor de combustión interna": ["mahai"]}
}
```

### Plain text

Any `.txt` file, UTF-8, paragraphs separated by blank lines. Line-break structure is reproduced exactly in the output.

### Glossary formats

A TSV, with `|` or `;` separating alternative targets:

```tsv
energía renovable	energia berriztagarri
red eléctrica	sare elektriko|elektrizitate-sare
```

Or a flat JSON object:

```json
{"energía renovable": "energia berriztagarri",
 "red eléctrica": ["sare elektriko", "elektrizitate-sare"]}
```

A flat glossary is loaded as the `proper` condition; `random` is left empty, and asking for `--mode random` then raises rather than silently running without terminology.

---

## Modes

| Mode | Terminology | Weights | Purpose |
|---|---|---|---|
| `proper` | correct entries | fine-tuned | the main condition |
| `random` | substituted targets | fine-tuned | control: does the system follow whatever it is handed? |
| `noterm` | none | fine-tuned | no terminology anywhere in the pipeline |
| `base_noterm` | none | base Latxa | contrastive: how much did fine-tuning buy? |

Repeat `--mode` to run several in one invocation; fine-tuned weights are scheduled first so the model pool reloads as little as possible.

```bash
python -m agenteak -d energy -m proper -m random -m noterm -i data/text.energy.eseu.json -t data/terms.energy.eseu.json
```

In `noterm` and `base_noterm` the matcher, the terminology agent and the term block in the editor's prompt are **all** inactive — the pipeline carries no terminology knowledge at all, not even the shape of a term list.

---

## How the terminology agent works

The matcher (`terminology.py`) is arithmetic, not a model: a glossary entry is present when every content token occurs **in order**, with only function words between them, and each token differs from the glossary form by at most a Spanish plural or gender ending. It is deliberately high-recall, and it makes two kinds of mistake the agent then fixes.

**False positives.** The words of a term co-occur inside a different noun phrase:

> *La **bomba de aceite** movida por el **motor**…*

`aceite de motor` is not used here, but its words are all present. (With the default `--term-max-gap 0` the matcher already rejects this one; raise the gap and it does not.)

**False negatives.** The mention is coordinated or reordered, so the tokens are not contiguous:

> *sistemas de frenado y de refrigeración*

`sistema de refrigeración` is genuinely used but never appears as a run. These are offered to the agent as *possible* entries — glossary entries sharing at least one content word with the chunk — and only a model reading the sentence can decide.

**Ambiguity.** Where an entry lists several approved targets, the agent picks one for this context.

The agent returns JSON. It is **reconciled** before use (`agents.py`): it may only return entries it was offered, and only targets that are approved. An invented term is dropped; an unapproved target falls back to the first approved one. An unparseable reply keeps the matcher's output, so a bad response degrades to the deterministic baseline rather than losing terminology.

Turn it off with `--no-term-agent` to run the matcher alone — faster, and a useful ablation.

### Few-shot examples

`prompts.py` carries three worked examples, one per decision type (reject / recover / disambiguate).

> ⚠️ **The Basque in these examples is illustrative.** It shows the agent the *shape* of each decision, not specific terminology. Before a submission run, have a Basque speaker check them or replace them with pairs from your own glossary — a wrong example is applied silently to every chunk.

---

## Prompts: what you may and may not edit

| Prompt | Status |
|---|---|
| Translator system prompt | 🔒 **frozen** — reproduces the fine-tuning data |
| Editor system + user prompts | 🔒 **frozen** — reproduces the fine-tuning data |
| Terminology agent prompt + few-shots | ✅ editable — that model is used zero-shot |

The frozen prompts include unaccented spellings in the editor's scaffolding (`declinacion`, `anadidos`). These are not typos to be tidied: the domain conditioning the fine-tune bought fires on these exact strings, and normalising them costs you the adaptation. The same applies to the `Ámbito:` line and the domain strings in `config.py`.

---

## Hardware

The profile is detected at startup and sets 4-bit loading, generation budgets, and how many 8B models stay resident.

| GPU | 4-bit | Resident 8B | Notes |
|---|---|---|---|
| A100 80GB | no | 2 | fastest; no reload between stages |
| A100 40GB | yes | 2 | |
| L4 / 24GB | yes | 1 | one 8B reload per stage per document |
| T4 16GB | yes | 1 | works |
| CPU | — | 1 | for testing only |

Override with `--4bit` / `--no-4bit`.

If both fine-tunes are published as LoRA adapters, both 8B roles can be served from **one** resident base model with an adapter switch instead of a reload:

```bash
python -m agenteak ... \
    --translator-adapter jurgiraud/latxa-eseu-wmt26-augmented-lora \
    --editor-adapter     jurgiraud/latxa-editor-wmt26-lora
```

---

## Output

```
output/
├─ agenteak.proper.automotive.eseu.json         # the submission artefact
├─ agenteak.proper.automotive.eseu.stats.json   # per-document statistics
├─ agenteak.proper.automotive.eseu.memory.json  # per-chunk store (--dump-memory)
└─ agenteak.proper.automotive.eseu.txt/         # text mode only
```

Both files are rewritten after **every** document, so an interrupted run costs at most the document in progress.

`stats.json` records, per document: chunk and sentence counts, candidate vs confirmed terms, how many the agent recovered and pruned, chunks edited, per-stage timings, and the diagnostics below.

### Diagnostics

Report-only. They gate nothing — the editor's output ships as it stands, and these exist so a run can be inspected without reading every segment.

| Type | Meaning |
|---|---|
| `numbers` | a figure in the source is missing from, or invented in, the translation |
| `omission` | target/source content-word ratio below 0.60 (collapse, not compression) |
| `repetition` | a phrase duplicated in place of another — garbled generation |
| `term_missing` | a selected term does not appear in the output |
| `term_form` | only the head noun of a multi-word term appears |

Plus a per-document stem-consistency check: the same term rendered on two different stems (`karteraren` beside `karterretik`) is worth seeing, and is only visible once the document is finished.

> These are **not** the terminology success rate reported in the paper. That is the official WMT25 scorer with our lemma-matching modifications, run separately on the finished submission.

`--dump-memory` writes source chunk, candidate terms, agent decision, draft and final side by side — the first thing to look at when output is wrong.

---

## Repository layout

```
agenteak/
├─ config.py        run configuration; domain strings; mode table
├─ textproc.py      normalisation, chunking, reconstruction, output hygiene
├─ terminology.py   deterministic matcher and glossary index
├─ agents.py        terminology agent reply parsing and reconciliation
├─ prompts.py       all prompts and few-shot examples
├─ models.py        model wrapper, thinking, LoRA switching, VRAM pool
├─ pipeline.py      graph nodes and LangGraph assembly
├─ diagnostics.py   report-only checks
├─ dataio.py        input/output for shared-task JSON and plain text
├─ runner.py        corpus driver
└─ cli.py           command line interface
examples/           small runnable sample data
tests/              offline tests (no GPU, no models)
```

`config.py`, `textproc.py`, `terminology.py`, `agents.py` and `dataio.py` import neither torch nor langgraph, so the logic that decides which terminology reaches the models can be tested on its own:

```bash
python -m tests.test_offline
```

---

## Reproducing the paper

```bash
for D in automotion energy; do
  python -m agenteak --domain $D \
      -m proper -m random -m noterm -m base_noterm \
      --input data/text.$D.eseu.json \
      --terms data/terms.$D.eseu.json \
      --out output/
done
```

Runs are seeded (`--seed`, default 42, offset per document), but generation is sampled at low temperature and exact reproduction across different hardware or transformers versions is not guaranteed.

---

## Citation

```bibtex

```

Please also cite the base model:

```bibtex
@inproceedings{sainz-etal-2025-instructing,
  title     = {Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque},
  author    = {Sainz, Oscar and Perez, Naiara and Etxaniz, Julen and Fernandez de Landa, Joseba
               and Aldabe, Itziar and García-Ferrero, Iker and Zabala, Aimar and Azurmendi, Ekhi
               and Rigau, German and Agirre, Eneko and Artetxe, Mikel and Soroa, Aitor},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  pages     = {29136--29160},
  url       = {https://aclanthology.org/2025.emnlp-main.1484/}
}
```

## Licence

Code released under the MIT licence (see `LICENSE`). The models it loads carry their own terms: Latxa and its fine-tunes are subject to the **Llama 3.1 Community License**, and Qwen3 to the **Apache 2.0** licence. Check both before redistributing weights or outputs.
