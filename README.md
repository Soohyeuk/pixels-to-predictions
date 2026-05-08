# Pixels to Predictions — DL Vision Challenge

[Competition Link](https://www.kaggle.com/competitions/pixels-to-predictions/overview)

P.S. The notebook is focused on running it on a Kaggle notebook, not on local level. To easily replicate, create a notebook in Kaggle, then import this ipynb file.

Fine-tuning `HuggingFaceTB/SmolVLM-500M-Instruct` for the ScienceQA-style visual multiple-choice competition. The model takes an image plus a question and choices, and predicts the correct 0-indexed answer.

## Approach

- **Base model:** SmolVLM-500M-Instruct loaded in fp16, frozen.
- **Adapters:** LoRA on attention + MLP projections across both the language and vision towers — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`, `fc1`, `fc2` (auto-inferred at runtime).
- **Rank:** `r=8`, `alpha=16`, dropout `0.05`. Trainable params ≈ 5.67M.
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
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Epochs | 2 |
| Learning rate | 3e-4 |
| Warmup ratio | 0.03 |
| Effective batch | 8 |
| Image size | 384 |
| Max text length | 1024 |

## Hardware

- Trained on a Kaggle **NVIDIA Tesla T4** (16 GB VRAM), `fp16`, gradient checkpointing on, batch 1 × grad-accum 8.
- Full training run (2 epochs, all training examples) takes ~3.5 h end-to-end on T4 (see `papermill` duration in notebook metadata).
- Peak VRAM ≈ 12 GB. CPU RAM ≈ 8 GB. Disk ≈ 2 GB for cached model + LoRA checkpoint.
- Should also fit on a single Colab T4 / L4. CPU-only inference works but is impractically slow.

## How to run

1. Run on Kaggle with the competition dataset attached at `/kaggle/input/competitions/pixels-to-predictions`. GPU must be enabled (T4 or better).
2. Execute cells top-to-bottom. The first run installs the packages listed in `requirements.txt` (also pinned inline in cell 1).
3. After `model.print_trainable_parameters()` runs, confirm trainable params are under 5M. If over, drop `LORA_R` to 4 in cell 2 and re-run.
4. The training cell calls `train_lora(train_df, max_train_examples=None)`. For a smoke test, change to `max_train_examples=64`.
5. Cells 4–5 score validation and write `submission.csv` to `/kaggle/working/`.

## Local / non-Kaggle setup

```bash
pip install -r requirements.txt
```

Then point `DATA_DIR` in cell 2 at your local copy of the competition CSVs (`train.csv`, `val.csv`, `test.csv`, plus the `images/` folder) and `OUT_DIR` / `CKPT_DIR` at writable paths.

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
