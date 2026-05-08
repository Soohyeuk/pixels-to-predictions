# Pixels to Predictions — DL Vision Challenge

Fine-tuning `HuggingFaceTB/SmolVLM-500M-Instruct` for the ScienceQA-style visual multiple-choice competition. The model takes an image plus a question and choices, and predicts the correct 0-indexed answer.

## Approach

- **Base model:** SmolVLM-500M-Instruct loaded in fp16, frozen.
- **Adapters:** DoRA (weight-decomposed LoRA) on attention + MLP projections — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Rank:** `r=4`, `alpha=16`, dropout `0.05`. Stays under the 5M trainable-parameter cap.
- **Scoring:** Multiple-choice via per-letter log-likelihood — we score `" A"`, `" B"`, … under the model and pick the lowest NLL. More robust than free-form generation.
- **Training target:** The label tokens cover only the final answer letter; the prompt, image tokens, and context are masked with `-100`.
- **Memory:** Gradient checkpointing on, batch size 1, grad accumulation 8 (effective batch ≈ 8).
- **Image size:** 384×384 (SmolVLM's native).

## Files

- `dl_competition.ipynb` — end-to-end pipeline: load data → attach adapters → train → log-likelihood eval → write `submission.csv` → diagnostic plots.
- `submission.csv` — produced in `/kaggle/working/` after running the final cells.

## Key hyperparameters

| Param | Value |
|---|---|
| LoRA rank | 4 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| DoRA | enabled |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Warmup ratio | 0.03 |
| Effective batch | 8 |
| Image size | 384 |
| Max text length | 1024 |

## How to run

1. Run on Kaggle with the competition dataset attached at `/kaggle/input/competitions/pixels-to-predictions`.
2. Execute cells top-to-bottom. The first run installs `transformers==4.57.6`, `peft==0.18.1`, `bitsandbytes`, `accelerate`, `datasets`, `pillow`.
3. After `model.print_trainable_parameters()` runs, confirm trainable params are under 5M. If over, drop `LORA_R` to 2 in cell 2 and re-run.
4. The training cell calls `train_lora(train_df, max_train_examples=None)`. For a smoke test, change to `max_train_examples=64`.
5. Cells 4–5 score validation and write `submission.csv` to `/kaggle/working/`.

## Constraints (per competition rules)

- Only the provided competition data — no external datasets.
- ≤5M trainable parameters.
- Offline evaluation (no network at scoring time).
- Must run within Kaggle / Colab free-tier resources.

## Submission format

`submission.csv` with two columns:

```
id,answer
test_02333,0
test_04102,2
```

`answer` is a 0-indexed integer valid for that row's choice set.
