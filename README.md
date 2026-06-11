# CineGraph — LightGCN Recommendation System
## Group 38 | Collaborative Filtering Assignment

> **Paper:** LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation  
> He et al., SIGIR 2020 — [DOI: 10.1145/3397271.3401063](https://doi.org/10.1145/3397271.3401063)

---

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (train everything, then open the app)
python -X utf8 run_all.py

# 3. Launch demo app (separate terminal)
streamlit run app/app.py
```

---

## Step-by-Step Commands

### Train LightGCN
```bash
python -X utf8 experiments/train_lightgcn.py
```

### Train LightGCN + Sentence-BERT (semantic init)
```bash
python -X utf8 experiments/train_lightgcn.py --semantic
```

### Train Baselines (MF, NCF, NGCF)
```bash
python -X utf8 experiments/train_baselines.py --model all
```

### Run Ablation Studies
```bash
python -X utf8 experiments/ablations.py --study all
```

### Cold-Start Evaluation
```bash
python -X utf8 experiments/cold_start.py
```

### Generate Final Results Table
```bash
python -X utf8 consolidate_results.py
```

### Verify Everything Works
```bash
python -X utf8 verify.py
```

### Launch Demo
```bash
streamlit run app/app.py
```

---

## Project Structure

```
CF  Prroject/
├── data/ml-1m/              # MovieLens-1M (auto-downloaded)
├── src/
│   ├── data_loader.py       # Preprocessing & BPR sampling
│   ├── graph.py             # Bipartite graph + normalised adjacency
│   ├── evaluate.py          # HR@K, NDCG@K metrics
│   ├── semantic_encoder.py  # Sentence-BERT item embeddings
│   ├── llm_agent.py         # Groq LLM (Llama-3.1-70B) agent
│   └── model/
│       ├── lightgcn.py      # Core LightGCN (SIGIR 2020)
│       ├── mf.py            # Baseline: Matrix Factorization
│       ├── ncf.py           # Baseline: Neural Collaborative Filtering
│       └── ngcf.py          # Baseline: NGCF
├── experiments/
│   ├── train_lightgcn.py    # LightGCN training script
│   ├── train_baselines.py   # Baseline training script
│   ├── ablations.py         # K and dim ablation studies
│   └── cold_start.py        # Cold-start evaluation
├── app/
│   ├── app.py               # Streamlit web UI
│   └── recommender_api.py   # Python API wrapper
├── checkpoints/             # Saved .pt model weights
├── results/                 # Metric JSON + plots + final_results.md
├── run_all.py               # Master pipeline script
├── verify.py                # Pre-demo verification script
├── consolidate_results.py   # Results table generator
├── demo.ipynb               # Jupyter walkthrough notebook
├── requirements.txt
└── .env                     # GROQ_API_KEY
```

---

## Model — LightGCN

LightGCN removes feature transformation and non-linear activations from GCN, retaining only neighbourhood aggregation:

**Propagation rule (no weight matrices, no activation):**
```
e_u^(k) = sum_{i in N_u}  (1 / sqrt(|N_u| |N_i|)) * e_i^(k-1)
```

**Final embedding — mean over all layers:**
```
e* = (1 / K+1) * sum_{k=0}^{K} e^(k)
```

**BPR Training objective:**
```
L = -sum ln sigmoid(y_ui - y_uj)  +  lambda * ||E^(0)||^2
```

---

## Extension — Semantic Initialisation

Item embeddings `E_item^(0)` are initialised with **Sentence-BERT** vectors encoded from movie title + genres, giving the model content-based knowledge from day 0. This particularly helps cold-start items with few interactions.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--emb_dim` | 64 | Embedding dimension |
| `--n_layers` | 3 | Graph conv layers (K) |
| `--lr` | 1e-3 | Adam learning rate |
| `--lambda_reg` | 1e-4 | L2 regularisation weight |
| `--batch_size` | 2048 | BPR mini-batch size |
| `--epochs` | 200 | Max training epochs |
| `--patience` | 15 | Early stopping patience (on NDCG@10) |

---

## Evaluation Protocol

- **Dataset:** MovieLens-1M (ratings >= 4 treated as positive)  
- **Split:** Leave-one-out (last interaction per user = test)  
- **Negative sampling:** 99 random negatives per test user  
- **Metrics:** HR@10, HR@20, NDCG@10, NDCG@20  

---

## Chat Demo (LLM Interface)

The Streamlit app includes an LLM chat powered by **Groq API (Llama-3.1-70B)**:

```
User: I'm user 42, what should I watch tonight?
CineGraph: Based on your history with action and sci-fi films, here are my top picks...

User: I love romantic comedies
CineGraph: Here are some great romantic comedies you might enjoy...
```

---

## References

1. He et al. (2020). LightGCN: Simplifying and Powering GCN for Recommendation. *SIGIR 2020*.
2. Wang et al. (2019). Neural Graph Collaborative Filtering. *SIGIR 2019*.
3. He et al. (2017). Neural Collaborative Filtering. *WWW 2017*.
4. Rendle et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI 2009*.
5. Reimers & Gurevych (2019). Sentence-BERT. *EMNLP 2019*.
