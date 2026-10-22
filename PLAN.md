# self_owned_llm — Hardening Plan

Written for: you, as the executor/author of this capstone.

Three improvements that turn the project from a convincing prototype into a
defensible one. Execute in order **2 → 1 → 3**: pin the environment first,
refactor SEAL eval into a shared core, then make the FastAPI service reuse that
same core so the CLI and the service can never drift apart again.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Item 2 — Pin every dependency version  (do first)

**Goal:** reproducible DGX runs; kill the TRL `max_seq_length` / Python 3.8-vs-3.9
failure class. **Constraint:** `torch` is tied to the node's CUDA build and is
likely preinstalled — pin to what's installed, never trigger a torch reinstall.

- [ ] On the DGX node where training actually runs (`python3.9` + `/jupyter`):
      `python3.9 -m pip freeze > requirements.lock.txt`
- [ ] Record `python3.9 --version` and `torch.__version__ / torch.version.cuda`
- [ ] Rewrite `requirements.txt` with `==` pins for every direct dep:
      torch, transformers, trl, peft, accelerate, datasets, tokenizers, numpy,
      pydantic, fastapi, uvicorn, pyyaml, sentencepiece, httpx, tqdm
- [ ] Commit both `requirements.txt` (direct `==`) and `requirements.lock.txt` (full freeze)
- [ ] README: document interpreter = `python3.9` and the exact run invocation;
      note `python3`/`python3.8` on the login node lacks torch
- [ ] `docker/Dockerfile`: pin Python 3.9/3.10 and install the lockfile; exclude
      the torch line if the base image ships its own CUDA torch

**Verify:** fresh venv → `pip install -r requirements.lock.txt` → all three
trainers import and run 1 step. Stamp versions into each adapter `meta.json`.
**Risk:** pinning torch breaks CUDA on the shared box → pin to the *installed*
version so pip is a no-op; never `--upgrade` torch without admin.

---

## Item 1 — Held-out SEAL evaluation (memorization → real generalization)

**The claim & a correction:** training on passage A then scoring on passage B
shows nothing improves — that's a *specificity/no-forgetting control*, not the
generalization claim. The real "it generalized, didn't memorize" claim is
**within-passage held-out QA**: adapt on *some* of passage A's Q&A, score on
*different, unseen* Q&A about passage A. Headline metric = within-passage held-out.

**Design**
- Primary (generalization): per passage, split `qa_pairs` into adapt vs held-out;
  train LoRA on adapt only; score base-vs-adapted token-F1 on held-out.
  The accept/reject gate switches to gating on **held-out** improvement.
- Control (specificity): a disjoint group of passages the adapter never touches;
  confirm adapted ≈ base there (no regression). This is the "passage B" check.
- Data reality: ~3 QA/passage is thin → use **k-fold within each passage** and
  report the **mean held-out improvement across all passages/folds**.

**Code**
- [ ] New `src/adaptation/seal_core.py` with reusable pieces lifted from
      `src/train/seal.py`: `load_passages()`, `split_qa(seed)`, `build_sample()`,
      `greedy_answer()`, `evaluate()`, `adapt_lora()`, `gate(base, adapted, margin)`.
      Reuse `src/common/metrics.py:token_f1` and `src/hitl/validators.py:RiskValidator`.
- [ ] Refactor `src/train/seal.py` to call `seal_core`; train on adapt-QA only;
      report `base_heldout`/`adapted_heldout`/`improvement_heldout` (headline) +
      `*_train` (sanity) + cross-passage `control_delta`; gate on held-out;
      write `runs/self_adapt/results.json` + adapter `meta.json`.
- [ ] `configs/seal_qwen.yaml`: add `eval_mode: within_passage|cross_passage|both`,
      `holdout_frac: 0.34` (or `kfold: 3`), `split_seed: 42`.

**Verify:** expect `improvement_train > improvement_heldout ≥ ~0` and
`control_delta ≈ 0`. If held-out ≈ 0 while train jumps, report that honestly
(it memorized). Stronger variant (future work): adapt on passage text /
self-generated restatements instead of QA strings (closer to real SEAL).
**Risk:** passages with <2 QA can't be split → skip and `log()` how many dropped.

---

## Item 3 — Unify on the Qwen track end-to-end (SEAL + FastAPI service)

**Goal:** one architecture in the live path: base = `Qwen/Qwen2.5-1.5B-Instruct`,
adaptations = **PEFT LoRA adapter directories** (not custom `.pt`), serving =
base + optional attached adapter. Retire `DecoderOnlyTransformer` from the service
path; keep its code as a documented offline "from-scratch learning track."

**Canonical contract**
- Load: `AutoModelForCausalLM.from_pretrained(base)` + `AutoTokenizer`.
- Adapt: `peft.get_peft_model(...)` → `model.save_pretrained(adapter_dir)`.
- Serve: base + `PeftModel.from_pretrained(base, adapter_dir)` (optional `merge_and_unload()`).
- Registry: reuse the `adapter_jobs` table in `src/hitl/db.py` + an `adapters/` dir.

**Refactors**
- [ ] `src/adaptation/adaptation_service.py` — rewrite `/adapt` to call `seal_core`:
      drop `Tokenizer.from_file` + `DecoderOnlyTransformer` + custom
      `inject_lora`/`save_lora_adapter`; use HF load + PEFT + `model.generate`;
      apply the **held-out gate**; save a PEFT adapter dir. Keep safety gate + token_f1.
- [ ] `configs/model_config.py` `PathConfig` — add `QWEN_BASE` + adapter dirs;
      mark `.pt` checkpoint constants deprecated for the Qwen path.
- [ ] `src/hitl/proposal_engine.py` — switch proposal generation to
      `AutoModel`/`AutoTokenizer` greedy (reuse `seal_core.greedy_answer`), making
      HITL fully Qwen: propose → RiskValidator → DB → approve → approved_lora_train.
- [ ] `services/model_service/app.py` + `src/api/gateway.py` — serve Qwen base +
      latest **accepted** adapter from the registry; add adapter select/rollback by `version_id`.
- [ ] From-scratch stack (`src/model/llm.py`, `src/lora/lora.py`, `src/train/pretrain.py`,
      `src/train/dpo.py`) — keep but mark "offline experimental, not wired into services."
- [ ] Tests: `test_peft_roundtrip` (save/attach Qwen adapter); integration test
      hitting `/adapt` on squad_mini asserting held-out improvement + HIGH-risk rejection.

**Verify (end-to-end on DGX):**
- CLI `seal.py` and the `/adapt` endpoint produce the **same** held-out numbers on
  the same input (proves no drift).
- HITL loop: generate (Qwen) → approve in DB → `src.train.hitl` → serve base+adapter
  → approved fact is answered.
- `grep -rn "DecoderOnlyTransformer\|Tokenizer.from_file" src/adaptation src/hitl services src/api`
  returns nothing (service path is Qwen-only).

**Risks:** VRAM for base+adapter (merge_and_unload / bf16); old `.pt` checkpoints
become dead for the service (document as deprecated, don't silently break);
biggest payoff *and* risk is the shared `seal_core` interface — get it right first.

---

## Cross-cutting
The shared `src/adaptation/seal_core.py` (Item 1) is what both the SEAL CLI and the
FastAPI service import in Item 3. It is the single fix for the "two half-systems"
problem — build and freeze its interface before wiring Item 3.

**Effort:** Item 2 ≈ 1 session · Item 1 ≈ 1–2 · Item 3 ≈ 2–3.


