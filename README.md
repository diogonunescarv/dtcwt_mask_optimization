# DT-CWT Mask Optimization (Protego-adapted)

User-centric facial privacy via a **Warm-Start** adversarial mask in the
**Dual-Tree Complex Wavelet (DT-CWT)** domain, built on the Protego codebase
([HKU-TASR/Protego](https://github.com/HKU-TASR/Protego)).

See [`PLAN.md`](PLAN.md) for the scientific motivation, Frequency Dispersion
Loss, and the full phased roadmap.

## Project layout

| Path | Role |
|------|------|
| `protego_base/` | Adapted Protego clone (FR models, training loop, optional UV/SMIRK) |
| `tests/` | Functional tests per development phase |
| `scripts/` | Shell runners |
| `notebooks/` | Explorations / EDA only |
| `PLAN.md` | Step-by-step development plan |

## Absolute paths (hardcoded convention)

| Purpose | Path |
|---------|------|
| Trained masks | `/mnt/study-data/dcarvalho/tests/` |
| Metrics / CSV / logs | `/mnt/study-data/dcarvalho/metrics/` |
| LFW frontal | `/mnt/databases/faces/lfw_frontal/{faces_train,faces_test}` |
| CelebA-HQ frontal | `/mnt/databases/faces/celebahq_pp_frontal/` |

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install a CUDA-matched torch build if needed, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Phase 1 — Protego cleanup & adaptation ✅

**Goal:** keep Protego’s FR loading + Gram-matrix training machinery, point
datasets at our absolute face roots (no identity pairs), and isolate the
3D UV / SMIRK stack behind `--original`.

### What changed

1. **Datasets** (`protego_base/protego/datasets.py`)
   - Flat loaders for `lfw_frontal` and `celebahq_pp_frontal`.
   - No multi-image identity grouping (unlike FaceScrub protectee folders).

2. **`--original` gate** (`train.py`, `run_exp.py`, `protego_train.py`, `UVMapping.py`)
   - **Default:** UV Mapping + SMIRK are **not** imported or initialized;
     masks apply as flat residuals (`three_d=False`, `bin_mask=False`).
   - **`--original`:** restores Protego’s UV warp + visibility masks (needs
     SMIRK / FLAME / MediaPipe assets from upstream `setup_quick.sh`).

3. **FR models** — unchanged entry point: `protego.FacialRecognition.FR`
   (ArcFace, AdaFace, FaceNet, MagFace, ViT, …). Weights download on first use.

### Run the Phase 1 functional test

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/protego_base${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/run_phase1_test.sh
# optional: skip FaceNet weight download
bash scripts/run_phase1_test.sh --skip-fr
```

The test checks dataset resolution, DataLoader batches, FR feature extraction,
and that UV/SMIRK stay unloaded without `--original`.

### Train entry (adapted)

```bash
cd protego_base
python train.py --dataset lfw_frontal --device cuda:0 --exp_name phase1_smoke --max_train 32 --max_eval 8
# Restore Protego UV/SMIRK pipeline:
python train.py --original --legacy_facescrub --device cuda:0
```

---

## Next phases (not started)

- **Phase 2:** Differentiable DT-CWT module + meta-mask on levels 2–3  
- **Phase 3:** Offline Frequency Dispersion Loss training  
- **Phase 4:** Online warm-start + short fine-tuning (`protect.py`)  
- **Phase 5:** Batch evaluation → `/mnt/study-data/dcarvalho/metrics/`
