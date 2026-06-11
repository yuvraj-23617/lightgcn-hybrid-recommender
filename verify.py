"""
verify.py
=========
Quick end-to-end verification script.
Run this before the demo to make sure everything works.

Usage
-----
    python verify.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import traceback

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(label, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
        results.append((label, True, ""))
    except Exception as e:
        msg = str(e).split('\n')[0]
        print(f"  {FAIL}  {label}  ->  {msg}")
        results.append((label, False, msg))


print("\n" + "="*60)
print("  CineGraph — System Verification")
print("="*60 + "\n")

# ── GPU ──────────────────────────────────────────────────────────────────── #
print("[ GPU ]")
def gpu_check():
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"         GPU: {name}  ({vram:.1f} GB VRAM)")
check("CUDA GPU available", gpu_check)

# ── Imports ───────────────────────────────────────────────────────────────── #
print("\n[ Imports ]")
check("data_loader",       lambda: __import__("src.data_loader",       fromlist=["ML1MDataset"]))
check("graph",             lambda: __import__("src.graph",             fromlist=["build_norm_adj"]))
check("evaluate",          lambda: __import__("src.evaluate",          fromlist=["evaluate"]))
check("model.lightgcn",    lambda: __import__("src.model.lightgcn",    fromlist=["LightGCN"]))
check("model.mf",          lambda: __import__("src.model.mf",          fromlist=["MatrixFactorization"]))
check("model.ncf",         lambda: __import__("src.model.ncf",         fromlist=["NCF"]))
check("model.ngcf",        lambda: __import__("src.model.ngcf",        fromlist=["NGCF"]))
check("semantic_encoder",  lambda: __import__("src.semantic_encoder",  fromlist=["encode_items"]))
check("llm_agent",         lambda: __import__("src.llm_agent",         fromlist=["LLMAgent"]))
check("recommender_api",   lambda: __import__("app.recommender_api",   fromlist=["RecommenderAPI"]))

# ── Dataset ───────────────────────────────────────────────────────────────── #
print("\n[ Dataset ]")
dataset = None
def load_dataset():
    global dataset
    from src.data_loader import ML1MDataset
    dataset = ML1MDataset()
    assert dataset.n_users > 0
    assert dataset.n_items > 0
    assert len(dataset.train_df) > 0
    print(f"         Users={dataset.n_users}  Items={dataset.n_items}  Train={len(dataset.train_df)}")
check("MovieLens-1M loads", load_dataset)

# ── Graph ─────────────────────────────────────────────────────────────────── #
print("\n[ Graph ]")
norm_adj = None
def build_graph():
    global norm_adj
    from src.graph import build_norm_adj
    device = torch.device("cuda")
    norm_adj, _ = build_norm_adj(dataset.train_df, dataset.n_users, dataset.n_items, device)
    assert norm_adj is not None
check("Bipartite graph construction", build_graph)

# ── Model ─────────────────────────────────────────────────────────────────── #
print("\n[ Model ]")
model = None
def build_model():
    global model
    from src.model.lightgcn import LightGCN
    device = torch.device("cuda")
    model = LightGCN(dataset.n_users, dataset.n_items, 64, 3, norm_adj).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"         Parameters: {n_params:,}")
check("LightGCN instantiation", build_model)

def forward_pass():
    device = torch.device("cuda")
    u  = torch.LongTensor([0,1,2]).to(device)
    pi = torch.LongTensor([10,11,12]).to(device)
    ni = torch.LongTensor([20,21,22]).to(device)
    bpr, reg = model(u, pi, ni)
    assert bpr.item() > 0
check("LightGCN forward pass", forward_pass)

# ── Checkpoint ────────────────────────────────────────────────────────────── #
print("\n[ Checkpoint ]")
def check_checkpoint():
    ckpt = "checkpoints/lightgcn_best.pt"
    assert os.path.exists(ckpt), f"Not found: {ckpt}"
    size_mb = os.path.getsize(ckpt) / 1024**2
    print(f"         {ckpt}  ({size_mb:.1f} MB)")
check("lightgcn_best.pt exists", check_checkpoint)

def load_checkpoint():
    from src.model.lightgcn import LightGCN
    device = torch.device("cuda")
    m = LightGCN(dataset.n_users, dataset.n_items, 64, 3, norm_adj).to(device)
    m.load_state_dict(torch.load("checkpoints/lightgcn_best.pt", map_location=device))
    m.eval()
check("Checkpoint loads successfully", load_checkpoint)

# ── LLM ───────────────────────────────────────────────────────────────────── #
print("\n[ LLM ]")
def check_groq():
    from src.llm_agent import LLMAgent
    agent = LLMAgent()
    reply = agent.chat("Say hello in 5 words or less.")
    assert len(reply) > 0
    print(f"         Groq reply: {reply[:80]}")
check("Groq API (LLM chat)", check_groq)

# ── Results ───────────────────────────────────────────────────────────────── #
print("\n[ Results Files ]")
for fname in ["lightgcn_results.json", "lightgcn_sbert_results.json",
              "baseline_results.json", "ablation_results.json"]:
    path = os.path.join("results", fname)
    exists = os.path.exists(path)
    status = PASS if exists else WARN
    print(f"  {status}  results/{fname}{'  (run training first)' if not exists else ''}")

# ── Summary ───────────────────────────────────────────────────────────────── #
print("\n" + "="*60)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"  Results: {passed}/{total} passed   {failed} failed")
if failed == 0:
    print("  Everything looks good! Run: streamlit run app/app.py")
else:
    print("  Some checks failed. Review errors above.")
print("="*60 + "\n")
