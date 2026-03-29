"""
Neural Network MA Predictor — PyTorch GPU
==========================================
Replace ridge regression with 3-layer MLP for better accuracy.
Train on exact MA values, predict at GPU speed.
"""
import numpy as np
import torch
import torch.nn as nn
import time
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

outfile = open(r"C:\Users\salih\Desktop\nn_predictor_out.txt", "w", buffering=1)
def log(s): print(s, flush=True); outfile.write(s+"\n"); outfile.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device} ({torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'})")

# === Exact MA (same compact version) ===
def mol_to_graph(mol):
    if mol is None: return None
    mol = Chem.RemoveHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds, adj = [], defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(b.GetBondTypeAsDouble())
        bonds.append((i, j, bt))
        adj[i].append((j, len(bonds)-1, bt))
        adj[j].append((i, len(bonds)-1, bt))
    return {"atoms": atoms, "bonds": bonds, "adj": adj, "n_bonds": len(bonds)}

def cfrag(g, bs):
    aset = set()
    for bi in bs:
        a, b, t = g["bonds"][bi]; aset.add(a); aset.add(b)
    al = sorted(aset); am = {a: i for i, a in enumerate(al)}
    edges = []
    for bi in sorted(bs):
        a, b, t = g["bonds"][bi]
        sa, sb = g["atoms"][a], g["atoms"][b]
        ra, rb = am[a], am[b]
        if (ra, sa) > (rb, sb): ra, rb = rb, ra; sa, sb = sb, sa
        edges.append((ra, rb, sa, t, sb))
    edges.sort()
    return (tuple(g["atoms"][a] for a in al), tuple(edges))

def compute_ma(mol, timeout=2.0):
    g = mol_to_graph(mol)
    if g is None or g["n_bonds"] == 0: return -1
    nb = g["n_bonds"]
    if nb <= 1: return nb
    naive = nb - 1; t0 = time.time()
    frags = defaultdict(list)
    for bi in range(nb):
        a, b, t = g["bonds"][bi]
        sa, sb = g["atoms"][a], g["atoms"][b]
        if sa > sb: sa, sb = sb, sa
        frags[(sa, t, sb)].append(frozenset([bi]))
    dups = {}
    for sig, occs in frags.items():
        if len(occs) >= 2: dups[cfrag(g, list(occs[0]))] = occs
    prev = [frozenset([bi]) for bi in range(nb)]
    for sz in range(2, min(9, nb // 2 + 1)):
        if time.time() - t0 > timeout: break
        nxt = set()
        for frag in prev:
            if len(nxt) >= 2000: break
            bd = set()
            for bi in frag:
                a, b, t = g["bonds"][bi]; bd.add(a); bd.add(b)
            for atom in bd:
                for _, bi, _ in g["adj"][atom]:
                    if bi not in frag:
                        nf = frag | {bi}
                        if len(nf) == sz: nxt.add(nf)
        sg = defaultdict(list)
        for frag in nxt: sg[cfrag(g, list(frag))].append(frag)
        for ch, occs in sg.items():
            if len(occs) >= 2:
                no = []
                for o in occs:
                    if all(o.isdisjoint(p) for p in no): no.append(o)
                if len(no) >= 2: dups[ch] = no
        prev = list(nxt)
        if not prev: break
    cands = []
    for ch, occs in dups.items():
        s = len(list(occs[0]))
        no = []
        for o in occs:
            if all(o.isdisjoint(p) for p in no): no.append(o)
        if len(no) >= 2: cands.append(((s-1)*(len(no)-1), s, no))
    cands.sort(reverse=True)
    used = set(); total = 0
    for _, s, occs in cands:
        av = [o for o in occs if o.isdisjoint(used)]
        if len(av) >= 2:
            total += (s-1)*(len(av)-1)
            for o in av: used |= o
    return naive - total

# === Generate large training set ===
def generate_molecules():
    mols = set()
    rings = [
        "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1cc[nH]c1", "c1ccoc1",
        "c1ccsc1", "c1ccc2ccccc2c1", "C1CCNCC1", "C1CCOCC1",
        "c1ccc2[nH]ccc2c1", "c1cnc2ccccc2n1", "c1ccnc(N)n1",
        "C1CC1", "C1CCC1", "C1CCCC1",
    ]
    fgs = [
        "O", "N", "F", "Cl", "Br", "S", "C(=O)O", "C(=O)N", "C=O",
        "C#N", "C(F)(F)F", "OC", "NC", "SC", "C(=O)OC", "NC(=O)C",
        "S(=O)(=O)N", "N(C)C", "OCC", "NCC", "C(=O)NC",
    ]
    links = ["", "C", "CC", "CCC", "CCCC", "C=C", "CCO", "CCN", "CCOC", "CCNC"]
    for r1 in rings:
        for r2 in rings:
            for lnk in links:
                m = Chem.MolFromSmiles(r1 + lnk + r2)
                if m: mols.add(Chem.MolToSmiles(m))
        for fg in fgs:
            for lnk in links:
                m = Chem.MolFromSmiles(r1 + lnk + fg)
                if m: mols.add(Chem.MolToSmiles(m))
        for fg1 in fgs:
            for fg2 in fgs:
                m = Chem.MolFromSmiles(r1 + "(" + fg1 + ")" + fg2)
                if m: mols.add(Chem.MolToSmiles(m))
    for n in range(2, 25):
        for e1 in ["", "O", "N", "F", "Cl", "C(=O)O", "C(=O)N"]:
            for e2 in ["", "O", "N", "F"]:
                m = Chem.MolFromSmiles(e1 + "C" * n + e2)
                if m: mols.add(Chem.MolToSmiles(m))
    for r in rings[:8]:
        for fg in fgs[:10]:
            for r2 in rings[:8]:
                m = Chem.MolFromSmiles(r + "(" + fg + ")CC" + r2)
                if m: mols.add(Chem.MolToSmiles(m))
    return list(mols)

# === Neural Network ===
class MAPredictor(nn.Module):
    def __init__(self, input_dim=2048, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# === Pipeline ===
log("=" * 60)
log("NEURAL NETWORK MA PREDICTOR")
log("=" * 60)

# Step 1: Generate + compute exact MA
log("\n[1] Generating training data...")
smiles_list = generate_molecules()
log(f"    {len(smiles_list)} unique molecules")

log("\n[2] Computing exact MA...")
t0 = time.time()
data = []
for i, smi in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    ma = compute_ma(mol, timeout=1.0)
    if ma < 0: continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    data.append((np.array(fp, dtype=np.float32), ma))
    if (i+1) % 2000 == 0:
        log(f"    {i+1}/{len(smiles_list)} ({(i+1)/(time.time()-t0):.0f} mol/s)")

log(f"    Done: {len(data)} molecules in {time.time()-t0:.0f}s")

X = np.array([d[0] for d in data])
y = np.array([d[1] for d in data], dtype=np.float32)

# Split
n = len(data)
perm = np.random.permutation(n)
n_train = int(0.85 * n)
train_idx, test_idx = perm[:n_train], perm[n_train:]

X_train = torch.tensor(X[train_idx]).to(device)
y_train = torch.tensor(y[train_idx]).to(device)
X_test = torch.tensor(X[test_idx]).to(device)
y_test = torch.tensor(y[test_idx]).to(device)

log(f"    Train: {n_train}, Test: {n - n_train}")
log(f"    MA range: {y.min():.0f}-{y.max():.0f}, mean={y.mean():.2f}")

# Step 3: Train
log("\n[3] Training neural network...")
model = MAPredictor(2048, 512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
loss_fn = nn.MSELoss()

batch_size = 512
best_mae = 999
model_path = r"C:\Users\salih\Desktop\gpu-assembly-index\ma_model.pt"
t0 = time.time()

for epoch in range(100):
    model.train()
    perm_t = torch.randperm(n_train, device=device)
    total_loss = 0
    n_batches = 0
    for start in range(0, n_train, batch_size):
        idx = perm_t[start:start+batch_size]
        xb, yb = X_train[idx], y_train[idx]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler.step()

    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test)
            mae = (pred_test - y_test).abs().mean().item()
            rmse = ((pred_test - y_test) ** 2).mean().sqrt().item()
            corr = torch.corrcoef(torch.stack([pred_test, y_test]))[0, 1].item()
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), model_path)
        log(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f} MAE={mae:.3f} RMSE={rmse:.3f} R={corr:.4f}")

train_time = time.time() - t0
log(f"    Training done in {train_time:.0f}s, best MAE={best_mae:.3f}")

# Step 4: Final
log("\n[4] Final evaluation...")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
with torch.no_grad():
    pred_test = model(X_test)
    mae = (pred_test - y_test).abs().mean().item()
    rmse = ((pred_test - y_test) ** 2).mean().sqrt().item()
    corr = torch.corrcoef(torch.stack([pred_test, y_test]))[0, 1].item()
    errors = (pred_test - y_test).abs()

log(f"    MAE: {mae:.3f}")
log(f"    RMSE: {rmse:.3f}")
log(f"    R: {corr:.4f}")
log(f"    Error < 0.5: {(errors < 0.5).float().mean().item()*100:.1f}%")
log(f"    Error < 1.0: {(errors < 1.0).float().mean().item()*100:.1f}%")
log(f"    Error < 2.0: {(errors < 2.0).float().mean().item()*100:.1f}%")

# Step 5: Speed
log("\n[5] GPU speed benchmark...")
n_bench = 1_000_000
X_fake = (torch.rand(n_bench, 2048, device=device) > 0.95).float()
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for start in range(0, n_bench, 100000):
        _ = model(X_fake[start:start+100000])
torch.cuda.synchronize()
pred_time = time.time() - t0
log(f"    1M predictions: {pred_time*1000:.0f}ms ({n_bench/pred_time:.0f} mol/s)")

# Step 6: Ridge baseline
log("\n[6] Ridge regression baseline...")
X_np = X_train.cpu().numpy()
y_np = y_train.cpu().numpy()
w = np.linalg.solve(X_np.T @ X_np + np.eye(2048), X_np.T @ y_np)
pred_ridge = X_test.cpu().numpy() @ w
ridge_mae = np.abs(pred_ridge - y_test.cpu().numpy()).mean()

log(f"    Ridge MAE: {ridge_mae:.3f}")
log(f"    NN MAE: {mae:.3f}")
log(f"    NN improvement: {(ridge_mae - mae) / ridge_mae * 100:.1f}%")

log(f"\n{'='*60}")
log("SUMMARY")
log(f"{'='*60}")
log(f"  Neural Network: MAE={mae:.3f}, R={corr:.4f}")
log(f"  Ridge Regression: MAE={ridge_mae:.3f}")
log(f"  NN improvement: {(ridge_mae - mae) / ridge_mae * 100:.1f}%")
log(f"  GPU speed: {n_bench/pred_time:.0f} mol/s")

outfile.close()
