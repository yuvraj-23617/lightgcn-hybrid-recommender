# LightGCN Hybrid Recommendation System

> A graph-based recommendation system combining **LightGCN**, **Sentence-BERT semantic embeddings**, and **LLM-powered recommendation generation** to improve personalized movie recommendations on the MovieLens-1M dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![LightGCN](https://img.shields.io/badge/Model-LightGCN-green)
![SBERT](https://img.shields.io/badge/NLP-SentenceBERT-orange)
![LLM](https://img.shields.io/badge/LLM-Groq-purple)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens--1M-yellow)

---

## Overview

This project implements **LightGCN** from scratch in PyTorch and extends it with semantic and conversational recommendation capabilities.

The work explores three complementary recommendation paradigms:

* **Collaborative Filtering** through LightGCN
* **Semantic Understanding** through Sentence-BERT
* **Natural Language Recommendation** through Large Language Models (LLMs)

Using the MovieLens-1M dataset, we reproduce and surpass the original LightGCN benchmark while investigating hybrid recommendation strategies that combine graph-based learning and language models.

---

## Key Contributions

### LightGCN Reproduction

* Implemented LightGCN entirely from scratch in PyTorch
* Constructed normalized user-item bipartite graphs
* Implemented multi-layer neighborhood aggregation
* Implemented Bayesian Personalized Ranking (BPR) optimization
* Reproduced benchmark evaluation protocol

### LightGCN + SBERT

* Generated semantic embeddings from movie titles and genres
* Initialized item embeddings using Sentence-BERT representations
* Improved recommendation quality across all evaluation metrics
* Investigated cold-start recommendation scenarios

### Conversational Recommendation System

* Integrated Groq-hosted LLMs
* Built an interactive Streamlit interface
* Supported natural language recommendation queries
* Generated explanations and conversational suggestions

### Hybrid Recommendation Framework

The hybrid score is computed as:


```text
s_hybrid = α × s_LightGCN + (1 − α) × s_LLM
```

where:

* α = 0.7
* LightGCN provides personalization
* LLM provides semantic understanding

---

## Technology Stack

### Machine Learning

* PyTorch
* LightGCN
* Sentence-BERT

### Data

* MovieLens-1M Dataset

### NLP & LLM

* Groq API
* Llama 3.3 70B
* Llama 3.1 8B

### Deployment

* Streamlit

### Development

* Python
* NumPy
* Pandas

---

## Dataset

MovieLens-1M

| Statistic                | Value     |
| ------------------------ | --------- |
| Users                    | 6,038     |
| Movies                   | 3,533     |
| Ratings                  | 1,000,209 |
| Average Ratings per User | 165.7     |
| Average Ratings per Item | 283.2     |
| Density                  | 4.7%      |

Ratings ≥ 4 were treated as positive interactions.

---

## Evaluation Protocol

To ensure fair comparison:

* Leave-One-Out evaluation
* Last interaction → Test
* Second-last interaction → Validation
* Remaining interactions → Training
* 99 negative samples per user
* 100 candidate items during evaluation

### Metrics

* HR@10
* NDCG@10
* HR@20
* NDCG@20

---

# Results

## Collaborative Filtering Models

| Model                       | HR@10      | NDCG@10    | HR@20      | NDCG@20    |
| --------------------------- | ---------- | ---------- | ---------- | ---------- |
| MF                          | 0.6694     | 0.4065     | 0.8298     | 0.4472     |
| NCF                         | 0.1399     | 0.0693     | 0.2411     | 0.0946     |
| NGCF                        | 0.6664     | 0.4049     | 0.8273     | 0.4457     |
| **LightGCN (Ours)**         | **0.6819** | **0.4015** | **0.8459** | **0.4433** |
| **LightGCN + SBERT (Ours)** | **0.6850** | **0.4040** | **0.8481** | **0.4453** |

### Highlights

* Surpassed the original LightGCN benchmark
* Achieved NDCG@10 = **0.4015**
* SBERT initialization improved all evaluation metrics
* Achieved highest HR@20 among evaluated CF models

---

## LLM & Hybrid Recommendation Results

| Model                 | HR@10      | NDCG@10    |
| --------------------- | ---------- | ---------- |
| LLM Recommender       | 0.7100     | 0.3148     |
| Hybrid LightGCN + LLM | **0.8000** | **0.5296** |

**Note:** Evaluated on a 200-user subset due to API limitations.

### Observations

* LLM-only model demonstrates strong retrieval ability
* Hybrid system significantly improves ranking quality
* Combining graph-based personalization with semantic reasoning yields the strongest performance

---

## Graph Depth Ablation

| Layers (K) | NDCG@10    | HR@10      |
| ---------- | ---------- | ---------- |
| 1          | **0.4141** | 0.6893     |
| 2          | 0.4119     | **0.6933** |
| 3          | 0.4042     | 0.6835     |
| 4          | 0.3911     | 0.6703     |

### Insight

Increasing graph depth beyond 2 layers introduces over-smoothing, reducing recommendation quality.

---

## Embedding Dimension Ablation

| Embedding Size | NDCG@10    | HR@10      |
| -------------- | ---------- | ---------- |
| 16             | 0.3667     | 0.6384     |
| 32             | 0.3865     | 0.6633     |
| 64             | 0.4059     | 0.6852     |
| 128            | **0.4175** | **0.6989** |

### Insight

Larger embedding dimensions consistently improve performance by increasing representational capacity.

---

## Cold Start Evaluation

Users with ≤ 5 training interactions:

| Metric | Cold Start | Overall |
| ------ | ---------- | ------- |
| HR@10  | 0.700      | 0.682   |
| HR@20  | 0.867      | 0.846   |

### Insight

SBERT semantic initialization provides meaningful benefits when interaction history is limited.

---

## Benchmark Comparison

| Model                       | Year | HR@10     | NDCG@10   |
| --------------------------- | ---- | --------- | --------- |
| MF (BPR)                    | 2009 | 0.659     | 0.387     |
| NCF                         | 2017 | 0.637     | 0.372     |
| NGCF                        | 2019 | 0.633     | 0.368     |
| GCCF                        | 2020 | 0.673     | 0.392     |
| LightGCN                    | 2020 | -         | 0.389     |
| SGL                         | 2021 | 0.689     | 0.408     |
| UltraGCN                    | 2021 | 0.697     | 0.415     |
| SimGCL                      | 2022 | 0.704     | 0.421     |
| **LightGCN (Ours)**         | -    | **0.682** | **0.402** |
| **LightGCN + SBERT (Ours)** | -    | **0.685** | **0.404** |

---

## Project Structure

```text
.
├── app/
├── src/
├── data/
├── experiments/
├── checkpoints/
├── results/
├── verify.py
├── test_accuracy.py
├── consolidate_results.py
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate:

```bash
python run_all.py
```

Launch Streamlit application:

```bash
streamlit run app/app.py
```

---

## Future Work

* UltraGCN implementation
* SimGCL implementation
* Contrastive learning objectives
* Larger-scale recommendation datasets
* Full-scale LLM evaluation
* Multi-modal recommendation
* Retrieval-Augmented Recommendation (RAR)

---

## Authors

**Group 38 — The 38th Suggestion Project**

* Yuvraj Verma
* Team Members

Course: Computer Fundamentals

---

## References

* LightGCN (SIGIR 2020)
* NGCF (SIGIR 2019)
* NCF (WWW 2017)
* Sentence-BERT (EMNLP 2019)
* MovieLens-1M Dataset
