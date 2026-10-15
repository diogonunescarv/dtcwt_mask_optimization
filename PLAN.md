# Projeto: Máscara Adversarial no Domínio DT-CWT com Perda de Dispersão de Frequência

## 1. Visão Geral e Contexto Científico
O objetivo deste projeto é desenvolver um método de proteção de privacidade facial focado no utilizador, capaz de gerar uma máscara adversarial **única e instantânea** no momento da submissão online, superando o gargalo de tempo de otimização de métodos como Fawkes, TIP-IM e LowKey.

A nossa arquitetura baseia-se num paradigma de **Warm-Start de Frequência**:
1. **Offline:** Treinamos uma "Meta-Máscara" universal no domínio das wavelets (DT-CWT).
2. **Online:** Aplicamos esta máscara na imagem e realizamos uma micro-alteração (ou pouquíssimos passos de otimização), tornando cada proteção única em milissegundos.

**Inspirações na Literatura:**
- **OPOM:** A ideia de afastar a imagem de um subespaço / distribuição e criar uma máscara base universal.
- **Chameleon:** A robustez a transformações de pipeline (compressão JPEG) garantida através da otimização das frequências corretas.
- **Protego (Baseline Arquitetural):** O conceito de Hipersensibilidade. Vamos utilizar o código base do Protego (`https://github.com/HKU-TASR/Protego`) para carregar modelos de Reconhecimento Facial (FR) e a matriz de Gram.
*Nota: O Protego utiliza deformação 3D via Mapa UV. Nós substituiremos o módulo de "Textura UV" por uma "Máscara no Domínio DT-CWT", mas manteremos a compatibilidade com o método original através da flag `--original`.*

---

## 2. A Matemática: Perda de Dispersão de Frequência (Frequency Dispersion Loss)
Em vez de modificar o espaço espacial (píxeis), operamos no espaço de frequências usando a Transformada Wavelet Complexa de Árvore Dupla (DT-CWT).
Seja $W_x$ os coeficientes da DT-CWT da face, e $\Delta W_{meta}$ a nossa Máscara complexa.

1. **Geração do Mini-lote Local (Offline):** Geramos $K$ versões da mesma imagem com micro-deslocamentos de fase $\phi_k$:
   $$W_{prot}^{(k)} = W_x + \Delta W_{meta} \cdot e^{i\phi_k}$$
   $$X_{prot}^{(k)} = \text{IDTCWT}(W_{prot}^{(k)})$$

2. **Perda de Dispersão:** Passamos essas imagens visualmente idênticas pela rede FR ($F$), gerando a Matriz de Gram $G$ para essas $K$ variações. Optimizamos a Meta-Máscara para maximizar a ortogonalidade (dispersar o espaço latente para aquela frequência):
   $$\mathcal{L}_{CWT-Dispersion} = - \log \det G_{u,v} \quad \text{onde} \quad G_{u,v} = F(X_{prot}^{(u)})^T \cdot F(X_{prot}^{(v)})$$

---

## 3. Diretrizes de Organização, Ambiente e Paths

### Ambiente Virtual e Dependências
1. Criar venv: `python -m venv .venv`
2. Ativar venv: `source .venv/bin/activate` (Linux/Mac) ou `.venv\Scripts\activate` (Windows).
3. Instalar o PyTorch e bibliotecas Wavelet (`pip install dtcwt pytorch-wavelets`).

### Regras de Estrutura de Diretórios e Git
- **Scripts de Terminal:** Qualquer automação de terminal deve ser criada como `.sh` e armazenada dentro da pasta `/scripts`.
- **Explorações e Prototipagem:** Devem ser feitas estritamente em notebooks dentro da pasta `/notebooks`.
- **Imagens:** **NÃO** salve imagens processadas ou originais no repositório. Atualize o `.gitignore` rigorosamente para ignorar `.jpg`, `.png`, e diretórios de dados locais.
- **Documentação:** A cada fase concluída, atualize o `README.md` principal do repositório ou crie um novo caso não exista, detalhando o progresso e como rodar o que foi feito.

### Caminhos Absolutos a Utilizar (Hardcoded/Config):
- **Caminho para salvar a Máscara treinada:** `/mnt/study-data/dcarvalho/tests/`
- **Caminho para salvar os Resultados (CSV/Logs):** `/mnt/study-data/dcarvalho/metrics/`
- **Datasets Base:** `/mnt/databases/faces/`
  1. `lfw_frontal/` (contém `faces_train` e `faces_test`)
  2. `celebahq_pp_frontal/`

---

## 4. Fases de Desenvolvimento Step-by-Step

*Atenção Agente: Prossiga na seguinte ordem. Só avance para o próximo passo após criar testes funcionais que comprovem o funcionamento da fase atual.*

### Fase 1: Limpeza e Adaptação do Repositório Base (Protego)
- [x] Ler a arquitetura do repositório clonado (`@Codebase`).
- [x] Identificar o carregamento dos modelos de FR (ex: ArcFace).
- [x] Adaptar o carregamento de Datasets para ler os nossos caminhos absolutos (`/mnt/databases/faces/lfw_frontal` e `celebahq_pp_frontal`), lembrando que não teremos pares da mesma imagem.
- [x] Refatorar o código isolando o "UV Mapping" e "SMIRK", mantendo-o acessível apenas caso o script seja chamado com a flag `--original`. Deixe comentários claros em cada parte desativada para entendimento futuro.
- [x] Criar o teste funcional desta fase.
- [x] Atualizar o `README.md`.

### Fase 2: Implementação do Módulo DT-CWT Diferenciável
- [x] Criar um módulo PyTorch que receba uma imagem, aplique o `DTCWTForward`, adicione um tensor de `meta_mask` (modificando magnitude e fase nos Níveis 2 e 3) e reconstrua com `DTCWTInverse`.
- [x] O Nível 1 da DT-CWT (altas frequências extremas) deve ser ignorado na otimização (congelado/zero) para evitar sensibilidade à compressão JPEG.
- [x] Criar o teste funcional para garantir a diferenciabilidade (se os gradientes fluem corretamente da imagem reconstruída até à máscara).
- [x] Atualizar o `README.md`.

### Fase 3: Treino Offline da Máscara Base (Meta-Máscara)
- [x] Implementar a `Frequency Dispersion Loss` (`-log(det(G))`).
- [x] Criar o script de treino: carregar um batch das bases LFW/CelebA-HQ e extrair a DT-CWT.
- [x] Geração do Lote Local: Somar a `meta_mask` e gerar $K$ variações aplicando minúsculas perturbações aleatórias na fase/magnitude.
- [x] Passar as variações pelo modelo FR, calcular a matriz de Gram e retropropagar os gradientes **apenas** para atualizar os tensores da `meta_mask`.
- [x] Salvar a máscara final otimizada em `/mnt/study-data/dcarvalho/tests/` (ex: `.pt` ou `.safetensors`).
- [x] Criar shell scripts em `/scripts` para rodar o treino.
- [x] Atualizar o `README.md`.

### Fase 3.1: Qualidade na Criação da Meta-Máscara (SSIM + L∞)
- [x] Adaptar os controlos de qualidade do Protego ao domínio DT-CWT (após IDT-CWT, sem UV/`bin_mask`).
- [x] Aplicar bound L∞ no residual espacial: \(X_{\text{prot}} = X + \mathrm{clamp}(\hat{X}-X, -\varepsilon, \varepsilon)\).
- [x] Adicionar SSIM hinge `max(min_ssim - SSIM, 0)` com peso adaptativo (igual ao Protego).
- [x] Combinar com a Frequency Dispersion Loss: \(\mathcal{L} = \mathcal{L}_{\text{disp}} + w \cdot \text{hinge}\).
- [x] Flags `--epsilon`, `--min_ssim`, `--quality` / `--no_quality`; runs versionadas `phase3_1`.
- [x] Criar testes funcionais e shell scripts em `/scripts`.
- [x] Atualizar o `README.md`.

### Fase 3.2: Identidade + Gram entre Batch + Ensemble multi-FR
- [x] Adicionar termo de identidade \(\mathrm{cos}(f_{\text{orig}}, f_{\text{prot}})\) (minimizar; menor = melhor desidentificação).
- [x] Adicionar Gram entre identidades do batch (\(-\log\det G_N\)), mantendo a dispersão K-fases.
- [x] Ensemble multi-FR como no Protego: média das losses por surrogate (BASIC_POOL sem `ir50_adaface_casia` por default).
- [x] Manter qualidade 3.1 (SSIM hinge + L∞) no caminho 3.2.
- [x] Dataset virtual `merged_lfw_celeba` (LFW train + CelebA-HQ).
- [x] Smoke test + script de treino grande (execução manual).
- [x] Atualizar o `README.md`.

### Fase 3.3: Identity ensemble + SSIM (sem dispersão-K / batch Gram)
- [x] Flag `--phase33`: \(\mathcal{L}=\overline{\cos}+\text{SSIM hinge}\) (+ L∞); sem `forward_k_variants` / batch Gram.
- [x] Split on-the-fly: todo `lfw_frontal/faces_train` + fração CelebA (`--celeba_train_frac`, `--data_seed`); eval = LFW test ∪ CelebA holdout.
- [x] Flag `--optimizer {adam,protego}` (Adam vs sign-SGD estilo Protego).
- [x] Ensemble default `fr_preset=small` (3 FRs); `full` disponível para hardware maior.
- [x] Smoke test (`scripts/run_phase3_3_test.sh`); treino full manual (`run_phase3_3_train*.sh`).
- [x] Notebook de comparação Adam vs Protego + atualizar `README.md`.

### Fase 4: Pipeline de Inferência Online (Warm-Start + Fine-Tuning Rápido)
- [ ] Criar o script de inferência (`protect.py`).
- [ ] Carregar a imagem do utilizador e o ficheiro da `meta_mask` do diretório `/mnt/study-data/dcarvalho/tests/`.
- [ ] Transformar a imagem (DT-CWT) e aplicar o **Warm-Start**: somar a `meta_mask` aos coeficientes dos níveis 2 e 3.
- [ ] **Otimização Online Rápida (Fine-Tuning):** Rodar um mini-loop de 2 a 5 iterações. Calcular o gradiente do modelo FR em relação aos coeficientes DT-CWT da imagem.
  - *Função de Perda:* Guiada por uma perda mista (inspirada no Fawkes/Chameleon): maximizar a distância do cosseno (ou L2) das *features* (desidentificação) + penalidade baseada no SSIM/DSSIM (qualidade visual).
- [ ] Reconstruir a imagem (IDTCWT) e retorná-la.
- [ ] Validar através de um script de teste funcional e atualizar o `README.md`.

### Fase 5: Testes em Massa e Extração de Métricas (Batch Evaluation)
- [ ] Criar o script de avaliação em lote (`evaluate_batch.py`) e um respetivo executável em `/scripts`.
- [ ] Carregar a partição de teste (ex: `faces_test` do `lfw_frontal`).
- [ ] Para cada imagem, executar o pipeline da Fase 4 e registrar:
  - **Qualidade Visual:** SSIM e LPIPS (comparando original vs protegida).
  - **Eficácia de Proteção:** Distância do cosseno e sucesso da camuflagem (Sim/Não).
  - **Tempo de Execução:** Tempo gasto por imagem.
- [ ] Salvar as métricas individuais e globais num ficheiro `.csv` no caminho `/mnt/study-data/dcarvalho/metrics/`.
- [ ] Validar os resultados nos `/notebooks` (apenas para EDA e visualização gráfica de métricas).
- [ ] Atualização final do `README.md`.