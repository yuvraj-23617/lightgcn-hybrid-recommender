# LightGCN — Final Results

**Dataset:** MovieLens-1M  
**Protocol:** Leave-one-out, 99 sampled negatives  

| Model | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|-------|------:|------:|------:|------:|
| MF | 0.6694 | 0.4065 | 0.8298 | 0.4472 |
| NCF | 0.1399 | 0.0693 | 0.2411 | 0.0946 |
| NGCF | 0.6664 | 0.4049 | 0.8273 | 0.4457 |
| LightGCN (ours) | 0.6819 | 0.4015 | 0.8459 | 0.4433 |
| LightGCN+SBERT (ours) | 0.6850 | 0.4040 | 0.8481 | 0.4453 |
| LLM Recommender (Groq) | 0.7100 | 0.3148 | 0.7800 | 0.3326 |
| Hybrid LightGCN+LLM (α=0.7) | **0.8000** | **0.5296** | **0.9300** | **0.5629** |

> Bold values = best per metric
