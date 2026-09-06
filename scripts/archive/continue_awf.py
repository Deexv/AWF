"""Continue training AWF LLM with more epochs to boost accuracy."""
import os, sys, time, math
sys.path.insert(0, "/home/z/my-project/awf")

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from core import AWFTransformer, num_params

DATA = "/home/z/my-project/data/corpus.txt"
OUT_DIR = "/home/z/my-project/download/AWF_LLM"
BLOCK = 64
BATCH = 32

with open(DATA) as f: text = f.read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, bs):
        self.data = encode(text); self.bs = bs
    def __len__(self): return max(0, len(self.data) - self.bs - 1)
    def __getitem__(self, i):
        c = self.data[i:i+self.bs+1]
        return torch.tensor(c[:-1]), torch.tensor(c[1:])

n_train = int(0.9 * len(text))
val_ds = TextDataset(text[n_train:], BLOCK)
val_loader = DataLoader(val_ds, batch_size=BATCH)
train_ds = TextDataset(text[:n_train], BLOCK)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

def evaluate(model, loader):
    model.eval()
    loss_sum, n, correct, total = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum")
            loss_sum += loss.item(); n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item(); total += y.numel()
    return loss_sum/n, correct/total

torch.manual_seed(0)
awf = AWFTransformer(vocab_size=59, d_model=64, n_layers=2, n_heads=4, block_size=BLOCK,
                     residual_rank=8, gen_kwargs=dict(n_fourier=16, hidden=96, n_layers=3), gen_grid=16)
awf.load_state_dict(torch.load(os.path.join(OUT_DIR, "awf_llm.pt")))
print(f"Loaded AWF: {num_params(awf):,} params")

# Current state
val_loss, val_acc = evaluate(awf, val_loader)
print(f"Before continuation: val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

opt = torch.optim.AdamW(awf.parameters(), lr=6e-4, weight_decay=0.01)
t0 = time.time()
for ep in range(10):
    awf.train()
    ep_loss, n = 0.0, 0
    for x, y in train_loader:
        opt.zero_grad()
        logits = awf(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(awf.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item() * x.size(0); n += x.size(0)
    val_loss, val_acc = evaluate(awf, val_loader)
    print(f"  ep{ep+1:02d} train={ep_loss/n:.4f} val={val_loss:.4f} acc={val_acc*100:.2f}% ({time.time()-t0:.0f}s)")
    if val_acc > 0.85:
        print("  Reached 85%+ val acc — stopping")
        break

torch.save(awf.state_dict(), os.path.join(OUT_DIR, "awf_llm.pt"))
print(f"Saved. Final val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
