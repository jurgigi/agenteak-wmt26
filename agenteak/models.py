"""Model layer: one wrapper per causal LM, plus a VRAM-aware pool.

Three models are needed per run (a 4B terminology agent and two 8B agents) and
on a 16GB card at 4-bit only two fit comfortably. Eviction is keyed on what is
actually LOADED, not on the role, so base_noterm — where translator and editor
are the same base weights — pays for one 8B model, not two, and a LoRA setup
pays for one base plus two adapter switches.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .config import MODE_SPECS


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

def detect_profile() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gb >= 70:
        return "a100_80"
    if gb >= 38:
        return "a100_40"
    if gb >= 20:
        return "l4"
    return "t4"


#  profile   -> 4bit,  think, answer, resident 8B models
PROFILES = {
    "a100_80": (False, 2048, 1536, 2),
    "a100_40": (True, 1536, 1280, 2),
    "l4":      (True, 1280, 1024, 1),
    "t4":      (True, 1024, 900, 1),
    "cpu":     (True, 512, 512, 1),
}


def resolve_budgets(cfg):
    """Fill in any budget the caller left as None from the detected profile."""
    profile = detect_profile()
    four_bit, think, answer, resident = PROFILES[profile]
    if cfg.load_in_4bit is None:
        cfg.load_in_4bit = four_bit
    if cfg.think_budget is None:
        cfg.think_budget = think
    if cfg.answer_budget is None:
        cfg.answer_budget = answer
    if cfg.max_resident_large is None:
        cfg.max_resident_large = resident
    return profile


def compute_dtype():
    return (torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16)


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------


class LMAgent:
    """Load, generate, optionally think, optionally switch LoRA adapters.

    Thinking support is DETECTED rather than assumed: the terminology agent is a
    Qwen3-family model and the two Latxa agents are Llama-3.1-based, so the
    chat-template contract differs per agent.
    """

    def __init__(self, model_id: str, name: str, adapter: Optional[str] = None,
                 adapter_name: Optional[str] = None, load_in_4bit: bool = True,
                 verbose: int = 1):
        self.name = name
        self.model_id = model_id
        self.verbose = verbose
        self.adapters = {}
        self.active_adapter = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"device_map": "auto", "attn_implementation": "sdpa"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype(),
                bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = compute_dtype()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except (ValueError, ImportError):        # sdpa unavailable for this arch
            kwargs.pop("attn_implementation")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            self.attach_adapter(adapter, adapter_name or "default")
        self.model.eval()

        tmpl = getattr(self.tokenizer, "chat_template", "") or ""
        self.supports_thinking = "enable_thinking" in tmpl
        self.think_end_id = self._think_end_id()
        self._warn_if_offloaded()

    # --- adapters ---------------------------------------------------------

    def attach_adapter(self, path: str, name: str):
        from peft import PeftModel
        if not self.adapters:
            self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name)
        elif name not in self.adapters:
            self.model.load_adapter(path, adapter_name=name)
        self.adapters[name] = path

    def use_adapter(self, name: Optional[str]):
        if not self.adapters or name == self.active_adapter:
            return
        if name is None:
            if hasattr(self.model, "disable_adapter_layers"):
                self.model.disable_adapter_layers()
        else:
            if hasattr(self.model, "enable_adapter_layers"):
                self.model.enable_adapter_layers()
            self.model.set_adapter(name)
        self.active_adapter = name

    # --- plumbing ---------------------------------------------------------

    def _think_end_id(self) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids("</think>")
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                return tid
        except Exception:
            pass
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return ids[-1] if len(ids) == 1 else None

    def _warn_if_offloaded(self):
        bad = ({str(d) for d in getattr(self.model, "hf_device_map", {}).values()}
               & {"cpu", "disk"})
        if bad:
            print(f"    *** WARNING: {self.name} has layers on {bad}; generation "
                  f"will be very slow.")
        elif self.verbose:
            print(f"    {self.name}: loaded ({self.model_id})")

    def _prompt(self, messages, thinking: bool = False) -> str:
        kw = {"enable_thinking": bool(thinking)} if self.supports_thinking else {}
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # --- generation -------------------------------------------------------

    def generate(self, messages, max_new_tokens=512, temperature=0.2,
                 top_p=0.9) -> str:
        prompt = self._prompt(messages, thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=temperature or None,
                top_p=top_p, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    def think(self, messages, think_budget=1024, answer_budget=700) -> tuple:
        """Thinking generation with a HARD cap -> (reasoning, answer).

        Phase 1 thinks up to think_budget; if </think> has not appeared we force
        it closed and continue, so total generation is bounded even when the
        model would happily reason forever.
        """
        prompt = self._prompt(messages, thinking=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=think_budget, do_sample=True,
                temperature=0.6, top_p=0.95, top_k=20,
                pad_token_id=self.tokenizer.eos_token_id)
        new = out[0][plen:].tolist()
        if self.think_end_id is not None and self.think_end_id in new:
            cut = len(new) - new[::-1].index(self.think_end_id)
            reasoning = self.tokenizer.decode(new[:cut], skip_special_tokens=True).strip()
            answer = self.tokenizer.decode(new[cut:], skip_special_tokens=True).strip()
            if answer:
                return reasoning, answer
        else:
            reasoning = (self.tokenizer.decode(new, skip_special_tokens=True).strip()
                         + " [budget reached]")
            close = self.tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
            out = torch.cat([out, torch.tensor([close], device=out.device)], dim=1)
        with torch.no_grad():
            final = self.model.generate(
                input_ids=out, attention_mask=torch.ones_like(out),
                max_new_tokens=answer_budget, do_sample=True,
                temperature=0.6, top_p=0.95, top_k=20,
                pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(final[0][out.shape[1]:],
                                       skip_special_tokens=True).strip()
        return reasoning, answer

    def ask(self, messages, thinking: bool = False, think_budget=1024,
            answer_budget=700) -> tuple:
        """Single entry point -> (reasoning, answer). Falls back to plain
        generation whenever thinking is off or unavailable."""
        if thinking and self.supports_thinking:
            return self.think(messages, think_budget, answer_budget)
        return "", self.generate(messages, max_new_tokens=answer_budget,
                                 temperature=0.2)


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class ModelPool:
    """Keeps as many models resident as the card allows and evicts the rest."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_large = max(1, cfg.max_resident_large or 1)
        self.cache = {}          # load_id -> LMAgent
        self.large = []          # load_ids, least-recently-used first

    def get(self, spec: dict) -> LMAgent:
        load_id, adapter, role = spec["load_id"], spec.get("adapter"), spec["role"]
        agent = self.cache.get(load_id)
        if agent is None:
            if spec.get("large", True):
                while len(self.large) >= self.max_large:
                    self._evict(self.large[0])
            if self.cfg.verbose:
                extra = f" + {adapter}" if adapter else ""
                print(f"    loading {role} ({load_id}{extra}) ...")
            agent = LMAgent(load_id, role, adapter=adapter,
                            adapter_name=role if adapter else None,
                            load_in_4bit=bool(self.cfg.load_in_4bit),
                            verbose=self.cfg.verbose)
            self.cache[load_id] = agent
            if spec.get("large", True):
                self.large.append(load_id)
        elif adapter:
            agent.attach_adapter(adapter, role)
        if load_id in self.large:
            self.large.remove(load_id)
            self.large.append(load_id)
        agent.use_adapter(role if adapter else None)
        return agent

    def _evict(self, load_id: str):
        agent = self.cache.pop(load_id, None)
        if load_id in self.large:
            self.large.remove(load_id)
        if agent is not None:
            if self.cfg.verbose:
                print(f"    evicting {agent.name} ({load_id}) to free VRAM")
            del agent.model
            del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self):
        for load_id in list(self.cache):
            self._evict(load_id)

    # --- role resolution --------------------------------------------------

    def role_spec(self, role: str, mode: str) -> dict:
        cfg = self.cfg
        weights = MODE_SPECS[mode]["weights"]
        if role == "term_agent":
            return {"role": "term_agent", "load_id": cfg.term_agent_model,
                    "adapter": None, "large": False}
        if weights == "base":
            return {"role": role, "load_id": cfg.base_model, "adapter": None,
                    "large": True}
        if role == "translator":
            if cfg.translator_adapter:
                return {"role": role, "load_id": cfg.base_model,
                        "adapter": cfg.translator_adapter, "large": True}
            return {"role": role, "load_id": cfg.translator_model,
                    "adapter": None, "large": True}
        if cfg.editor_adapter:
            return {"role": role, "load_id": cfg.base_model,
                    "adapter": cfg.editor_adapter, "large": True}
        return {"role": role, "load_id": cfg.editor_model, "adapter": None,
                "large": True}

    def agent_for(self, role: str, mode: str) -> LMAgent:
        return self.get(self.role_spec(role, mode))
