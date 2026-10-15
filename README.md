# DT-CWT Mask Optimization (Protego-adapted)

Proteção de privacidade facial **centrada no utilizador**: uma **meta-máscara universal** no domínio da **Dual-Tree Complex Wavelet (DT-CWT)**, treinada offline e aplicada em tempo quase constante (warm-start). O código parte do [Protego](https://github.com/HKU-TASR/Protego) (CVPR 2026) — carregamento de FRs, Gram / hipersensibilidade, controlos de qualidade — substituindo a textura UV 3D por coeficientes wavelet.

Roadmap científico e fases: [`PLAN.md`](PLAN.md).

---

## Ideia em resumo

| | Detalhe |
|---|---|
| **O que se aprende** | `delta_mag` / `delta_phase` **espaciais** nos níveis DT-CWT **2 e 3** (nível **1 congelado**, robustez JPEG) |
| **Como se aplica** | \(W' = W + \Delta W_{\mathrm{meta}}\cdot e^{i\phi}\) → IDT-CWT → residual com **L∞** \(\varepsilon=16/255\) |
| **Offline** | Uma máscara para muitas faces (LFW + CelebA-HQ) |
| **Online (Fase 4, TBD)** | Aplicar a máscara + poucos passos de fine-tune |
| **Métrica de desID** | \(\mathrm{cos}(f_{\mathrm{orig}}, f_{\mathrm{prot}})\) — **menor = melhor** |
| **Qualidade** | SSIM hinge (\(\texttt{min\_ssim}=0.95\)) + L∞; LPIPS na eval |

Contraste rápido com linhas paralelas do mesmo autor:

- **[Lightweight Face De-identification](https://github.com/diogonunescarv/Lightweight-Face-De-identification):** transformador leve (DCT/DT-CWT **global** por banda + flow/photo), hinge até `target_cos`, Adam.
- **[fawkes-dtcwt](https://github.com/diogonunescarv/fawkes-dtcwt):** cloak **por imagem** (otimiza \(\Delta M,\Delta\Phi\) densos na proteção).
- **Este repo:** máscara **universal espacial** L2/L3 + loss estilo Protego (identidadе / dispersão / ensemble FR).

---

## Melhores resultados até agora

### Campeão atual — Phase 3.3 + otimizador **Protego (sign-SGD)**

Run: `20261015_055232_phase3_3_merged_p33_protego`  
Setup: `--phase33`, `fr_preset=small` (3 FRs), `celeba_train_frac=0.5`, `data_seed=42`, 5 epochs, \(\varepsilon=16/255\), `min_ssim=0.95`.  
Loss: média do cos no ensemble + SSIM hinge (sem dispersão-\(K\) / batch Gram).

**Eval ampliada (não vista no treino)** — notebook  
[`notebooks/phase3_3_visualize_merged_identity_mask.ipynb`](notebooks/phase3_3_visualize_merged_identity_mask.ipynb) §6:

| | Valor |
|---|---|
| Conjunto | **2029** LFW `faces_test` + **2029** CelebA holdout (**4058** total) |
| FR de eval | `ir50_arcface_casia` |
| **% cos &lt; 0.5** | **84.1%** |
| cos médio / mediano | **0.327** / 0.308 |
| SSIM médio | 0.935 |
| LPIPS médio (Alex) | **0.118** |
| Tempo | ~96 s (~42 img/s, GPU) |

Por fonte:

| Fonte | n | cos médio | % cos&lt;0.5 | SSIM | LPIPS |
|-------|---|-----------|-------------|------|-------|
| CelebA holdout | 2029 | 0.298 | **90.3%** | 0.943 | **0.059** |
| LFW test | 2029 | 0.357 | 77.8% | 0.927 | 0.176 |

**Leitura:** desidentificação forte com qualidade perceptual aceitável (LPIPS ~0.12). CelebA fica melhor em qualidade **e** desID; LFW é o gargalo visual (LPIPS ~3× maior). Quase nenhum frame cumpre SSIM ≥ 0.95 (`frac≈9.6%`), mas LPIPS indica que a face não está destruída — o hinge 0.95 é mais rígido que a percepção.

### Adam vs sign-SGD (mesmo split / epochs / dados)

| | Adam (`lr=0.01`) | **Protego sign-SGD** |
|---|---|---|
| cos final (treino) | ≈ 0.999 | ≈ **0.35** |
| cos médio (n=200 eval) | ≈ 0.998 | ≈ **0.28** |
| % cos &lt; 0.5 (n=200) | **0%** | **92.5%** |
| SSIM médio (n=200) | 0.970 | 0.941 |

Adam colapsa na qualidade (máscara quase inerte). Sign-SGD (estilo PGD/Protego) força a máscara a usar o orçamento e reduz cos de forma consistente. **Usar `--optimizer protego` para Phase 3.3.**

### Phase 3.2 (dispersão-K + batch Gram) — referência

Run típico `merged_K4` (`max_train=4096`, K=4): cos médio ~**0.78** em 200 LFW test, mas SSIM ~**0.83** (abaixo do alvo 0.95; `frac SSIM≥0.95 = 0`). Melhor desID relativa que Adam-3.3, pior qualidade que Protego-3.3 no eval grande. A dispersão ajuda o ataque; sem calibragem forte de SSIM/L∞ o trade-off fica desfavorável.

---

## Layout do repositório

| Path | Papel |
|------|------|
| `protego_base/` | Clone Protego adaptado (FR, datasets, `train_meta_mask.py`, DT-CWT) |
| `protego_base/protego/dtcwt_mask.py` | Módulo diferenciável da meta-máscara |
| `protego_base/protego/dispersion_loss.py` | \(-\log\det G\) (variantes K / batch) |
| `protego_base/protego/identity_loss.py` | Cosine identidade (minimizar) |
| `protego_base/protego/quality_loss.py` | L∞ + SSIM hinge + peso adaptativo |
| `tests/` | Testes funcionais por fase |
| `scripts/` | Runners shell (smoke + treino manual) |
| `notebooks/` | Visualização / comparação de runs |
| `PLAN.md` | Plano científico e checklist de fases |

Pesos de FR **não** vão para o Git: ficam em `protego_base/FR_DB/*/pretrained/` (gitignore). Máscaras e métricas ficam em `/mnt/study-data/...`.

---

## Paths absolutos (convenção)

| Uso | Path |
|-----|------|
| Máscaras / checkpoints | `/mnt/study-data/dcarvalho/tests/` |
| Métricas / CSV / logs | `/mnt/study-data/dcarvalho/metrics/` |
| LFW frontal | `/mnt/databases/faces/lfw_frontal/{faces_train,faces_test}` |
| CelebA-HQ frontal | `/mnt/databases/faces/celebahq_pp_frontal/` |

Cada run cria `YYYYMMDD_HHMMSS_phaseX_<exp>/` + pointer `LATEST_phaseX_<exp>.txt`.

---

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Torch CUDA alinhado à máquina, se preciso:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
export PYTHONPATH="$PWD/protego_base${PYTHONPATH:+:$PYTHONPATH}"
```

Dependências relevantes: `pytorch-wavelets`, `pytorch-msssim`, `lpips`, FRs via código Protego (download no 1.º uso).

---

## Fases implementadas

### Phase 1 — Adaptação Protego ✅

- Datasets flat (sem pastas por identidade).
- UV/SMIRK só com `--original`; default = residual flat.
- FR: `protego.FacialRecognition.FR` (ArcFace, AdaFace, FaceNet, …).

```bash
bash scripts/run_phase1_test.sh
```

### Phase 2 — DT-CWT diferenciável ✅

`DTCWTMetaMaskModule`: forward → máscara L2/L3 → inverse; L1 sem parâmetros; grads até `delta_mag` / `delta_phase`.

```bash
bash scripts/run_phase2_test.sh
```

### Phase 3 — Dispersão de frequência ✅

\(\mathcal{L}=-\log\det G\) sobre \(K\) variantes de fase da mesma face.

```bash
bash scripts/run_phase3_test.sh
bash scripts/run_phase3_train.sh --exp_name lfw_K4 --epochs 3 --max_train 512 --K 4
```

### Phase 3.1 — Qualidade (SSIM + L∞) ✅

Após IDT-CWT: clamp L∞ + hinge SSIM com peso adaptativo \(w\).

```bash
bash scripts/run_phase3_1_test.sh
bash scripts/run_phase3_1_train.sh --exp_name lfw_K4_q --epochs 3 --max_train 512 --K 4
```

### Phase 3.2 — Identidade + Gram batch + ensemble ✅

\[
\mathcal{L}=\overline{\mathcal{L}}_K+\overline{\mathcal{L}}_{\mathrm{batch}}+\overline{\cos}+w\cdot\mathrm{hinge}_{\mathrm{SSIM}}
\]

Preset `small`: `ir50_arcface_casia`, `mobilefacenet_arcface_casia`, `ir18_adaface_webface`.

```bash
bash scripts/run_phase3_2_test.sh
bash scripts/run_phase3_2_train.sh --exp_name merged_K4 --epochs 3 --max_train 4096 --K 4
```

### Phase 3.3 — Identidade + SSIM (sem K-disp) ✅ **recomendado**

\[
\mathcal{L}=\frac{1}{|\mathcal{E}|}\sum_e \cos\big(F_e(X),F_e(X_{\mathrm{prot}})\big)
+w\cdot\max(\texttt{min\_ssim}-\mathrm{SSIM},0)
\]

Split: LFW train + fração CelebA (`--celeba_train_frac`, `--data_seed`); eval = LFW test ∪ holdout CelebA.

| `--optimizer` | Update | LR default |
|---------------|--------|------------|
| `adam` | Adam | `1e-2` |
| **`protego`** | `param -= lr * sign(grad)` | `0.01 * epsilon` |

```bash
bash scripts/run_phase3_3_test.sh
OPTIMIZER=protego bash scripts/run_phase3_3_train.sh
bash scripts/run_phase3_3_train_compare.sh   # Adam depois Protego
```

Resume / tmux:

```bash
bash scripts/run_phase3_3_train.sh \
  --resume /mnt/study-data/dcarvalho/tests/<run_id>/checkpoint_latest.pt

tmux new -s p33
OPTIMIZER=protego bash scripts/run_phase3_3_train.sh
```

Notebooks úteis:

| Notebook | Conteúdo |
|----------|----------|
| `phase3_3_visualize_merged_identity_mask.ipynb` | Adam vs Protego + **eval grande** (cos/SSIM/LPIPS) |
| `phase3_3_visualize_merged_p33_adam.ipynb` | Só run Adam |
| `phase3_2_visualize_merged_K4_mask.ipynb` | Máscara 3.2 K=4 |

---

## Como interpretar métricas

| Métrica | Direção | Nota |
|---------|---------|------|
| `cos_similarity` | ↓ melhor | Limiar prático de “desID” usado aqui: **0.5** |
| SSIM | ↑ melhor | Alvo treino 0.95 (hinge); na prática Protego-3.3 fica ~0.93–0.94 |
| LPIPS (Alex) | ↓ melhor | ~0.06 CelebA / ~0.18 LFW no melhor run |
| L∞ | ≤ ε | Sempre após IDT-CWT via `apply_linf_budget` |

---

## Próximas fases

- **Phase 4:** `protect.py` — warm-start da meta-máscara + 2–5 passos online.
- **Phase 5:** `evaluate_batch.py` — métricas em massa (cos, SSIM, LPIPS, tempo) → `/mnt/study-data/dcarvalho/metrics/`.

---

## Git / artefactos pesados

Ignorados (raiz + `protego_base/.gitignore`): pesos `FR_DB/*/pretrained/`, `FD_DB/*/pretrained/`, SMIRK assets, `*.pt`/`*.pth`/`*.ckpt`, imagens, CSV/logs de experiência. **Não** versionar checkpoints de máscara nem bases de faces — só código, scripts, testes e notebooks.
