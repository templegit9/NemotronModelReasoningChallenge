# NVIDIA Nemotron Model Reasoning Challenge — Task Tracker

## Legend
- **Owner**: `YOU` = user action on Kaggle, `CLAUDE` = Claude builds locally, `BOTH` = collaborative
- **Status**: `DONE` | `NEXT` | `BLOCKED` | `READY` | `IN PROGRESS`

---

## Phase 1: Local Foundation
| # | Task | Owner | Status | Blocked By | Notes |
|---|------|-------|--------|------------|-------|
| 1 | categorize_data.py — classify & split data | CLAUDE | DONE | — | 9,500 rows → 6 categories, 90/10 split (8,553/947) |
| 2 | Build algorithmic solvers (6 categories) | CLAUDE | DONE | — | 100% on roman/physics/unit, 99.9% encryption, 8.7% bit, 0.7% symbol |
| 3 | verify_solvers.py — validate against training data | CLAUDE | DONE | — | Overall 68.4% solvable algorithmically |
| 4 | Chat template investigation notebook | CLAUDE | DONE | — | notebooks/investigate_chat_template.py ready to run |

## Phase 2: Kaggle Pipeline Setup
| # | Task | Owner | Status | Blocked By | Notes |
|---|------|-------|--------|------------|-------|
| 5 | ~~Run chat template investigation on Kaggle~~ | YOU | DONE | — | Template uses `<\|im_start\|>/<\|im_end\|>`, `<think>/<​/think>` tags, empty system msg |
| 6 | **Submit untrained baseline adapter** | YOU | NEXT | — | Fork starter notebook, run as-is, submit for baseline score |
| 7 | ~~Format training data with CoT reasoning~~ | CLAUDE | DONE | #5 | 8,553 train + 947 val formatted as JSONL with CoT |
| 8 | ~~Build Kaggle training notebook~~ | CLAUDE | DONE | #7 | notebooks/train_submission.py — LoRA fine-tuning with CoT data |
| 9 | **Run first trained submission on Kaggle** | YOU | BLOCKED | #8, #14 | Upload training notebook, run, submit, share score |
| 14 | **Verify identity on Kaggle for submission** | YOU | NEXT | — | Required by competition — do ASAP |

## Phase 3: Autoresearch Integration
| # | Task | Owner | Status | Blocked By | Notes |
|---|------|-------|--------|------------|-------|
| 10 | Set up autoresearch loop (local proxy model) | CLAUDE | **NEXT** | #7 | program.md, train.py, evaluate.py for RTX 5070 Ti |
| 11 | **Run autoresearch overnight experiments** | YOU | BLOCKED | #10 | Launch locally, ~12 experiments/hour, review next morning |

## Phase 4: Advanced Optimization
| # | Task | Owner | Status | Blocked By | Notes |
|---|------|-------|--------|------------|-------|
| 12 | Generate synthetic training data (2x augmentation) | CLAUDE | READY | — | Roman, unit, physics, encryption, bit, symbol |
| 13 | Iterate: optimize per-category accuracy | BOTH | BLOCKED | #9 | CoT tuning, sampling weights, hyperparams, daily Kaggle submissions |

---

## Current Priority
1. **Task #14** [YOU] — Verify your identity on Kaggle (required before any submission)
2. **Task #6** [YOU] — Submit untrained baseline adapter on Kaggle
3. **Task #9** [YOU] — Run `notebooks/train_submission.py` on Kaggle and submit (needs #14 done first)
4. **Task #10** [CLAUDE] — Set up autoresearch loop with local proxy model (NEXT)

---

## Competition Info
- **Prize:** $106,388 USD
- **Deadline:** 2026-06-15 (merger deadline: 2026-06-08)
- **Max daily submissions:** 5
- **Max team size:** 5
- **GPU:** NVIDIA RTX Pro 6000 on Kaggle
- **Model:** Nemotron-3-Nano-30B-A3B (3B active, hybrid Mamba-Transformer MoE)
- **Submission:** LoRA adapter (rank ≤ 32), evaluated via vLLM with temperature=1.0

## Estimated Accuracy Targets
| Category | Target | Current |
|----------|--------|---------|
| Number System | ~95%+ | — |
| Unit Conversion | ~90%+ | — |
| Physics | ~85%+ | — |
| Text Encryption | ~75%+ | — |
| Bit Manipulation | ~45%+ | — |
| Symbol Transform | ~45%+ | — |
| **Weighted Total** | **~72%** | **—** |
