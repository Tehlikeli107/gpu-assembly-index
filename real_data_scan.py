"""
Real-World Drug Data Scan
=========================
Download REAL molecular datasets and compute Assembly Index.
- MoleculeNet ESOL (solubility)
- MoleculeNet Lipophilicity
- MoleculeNet BBBP (blood-brain barrier)
- FDA approved drugs

DISCOVERY TARGET: Does MA correlate with drug properties?
If MA predicts solubility/lipophilicity/BBB penetration -> practical value!
"""
import numpy as np
import torch
import time
import csv
import urllib.request
import io
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

outfile = open(r"C:\Users\salih\Desktop\real_data_out.txt", "w", buffering=1)
def log(s): print(s, flush=True); outfile.write(s+"\n"); outfile.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Compact MA ===
def compute_ma(mol, timeout=2.0):
    mol2 = Chem.RemoveHs(mol)
    atoms = [a.GetSymbol() for a in mol2.GetAtoms()]
    bonds, adj = [], defaultdict(list)
    for b in mol2.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(b.GetBondTypeAsDouble())
        bonds.append((i, j, bt))
        adj[i].append((j, len(bonds)-1, bt))
        adj[j].append((i, len(bonds)-1, bt))
    nb = len(bonds)
    if nb <= 1: return nb
    def cfrag(bs):
        aset = set()
        for bi in bs:
            a, b, t = bonds[bi]; aset.add(a); aset.add(b)
        al = sorted(aset); am = {a: i for i, a in enumerate(al)}
        edges = []
        for bi in sorted(bs):
            a, b, t = bonds[bi]
            sa, sb = atoms[a], atoms[b]
            ra, rb = am[a], am[b]
            if (ra, sa) > (rb, sb): ra, rb = rb, ra; sa, sb = sb, sa
            edges.append((ra, rb, sa, t, sb))
        edges.sort()
        return (tuple(atoms[a] for a in al), tuple(edges))
    naive = nb - 1; t0 = time.time()
    frags = defaultdict(list)
    for bi in range(nb):
        a, b, t = bonds[bi]
        sa, sb = atoms[a], atoms[b]
        if sa > sb: sa, sb = sb, sa
        frags[(sa, t, sb)].append(frozenset([bi]))
    dups = {}
    for sig, occs in frags.items():
        if len(occs) >= 2: dups[cfrag(list(occs[0]))] = occs
    prev = [frozenset([bi]) for bi in range(nb)]
    for sz in range(2, min(9, nb // 2 + 1)):
        if time.time() - t0 > timeout: break
        nxt = set()
        for frag in prev:
            if len(nxt) >= 2000: break
            bd = set()
            for bi in frag:
                a, b, t = bonds[bi]; bd.add(a); bd.add(b)
            for atom in bd:
                for _, bi, _ in adj[atom]:
                    if bi not in frag:
                        nf = frag | {bi}
                        if len(nf) == sz: nxt.add(nf)
        sg = defaultdict(list)
        for frag in nxt: sg[cfrag(list(frag))].append(frag)
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

# === Download MoleculeNet datasets ===
def download_csv(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AssemblyGPU/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)
    except Exception as e:
        log(f"  Download error: {e}")
        return []

def process_dataset(name, url, smiles_col, target_col, target_name):
    """Download dataset, compute MA, analyze correlation with target property."""
    log(f"\n{'='*60}")
    log(f"DATASET: {name}")
    log(f"{'='*60}")

    rows = download_csv(url)
    if not rows:
        log("  Failed to download")
        return None

    log(f"  Downloaded {len(rows)} molecules")

    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        smi = row.get(smiles_col, "")
        target_val = row.get(target_col, "")
        if not smi or not target_val:
            continue
        try:
            target_val = float(target_val)
        except ValueError:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        ma = compute_ma(mol, timeout=2.0)
        if ma < 0: continue

        nb = mol.GetNumBonds()
        na = mol.GetNumHeavyAtoms()
        mw = Descriptors.ExactMolWt(mol)

        results.append({
            "smiles": smi, "ma": ma, "bonds": nb, "atoms": na,
            "mw": mw, "target": target_val
        })

        if (i+1) % 200 == 0:
            log(f"    {i+1}/{len(rows)} ({(i+1)/(time.time()-t0):.0f} mol/s)")

    elapsed = time.time() - t0
    log(f"  Processed: {len(results)} molecules in {elapsed:.0f}s")

    if len(results) < 10:
        log("  Too few results")
        return None

    # GPU analysis
    ma_arr = torch.tensor([r["ma"] for r in results], dtype=torch.float32, device=device)
    target_arr = torch.tensor([r["target"] for r in results], dtype=torch.float32, device=device)
    bonds_arr = torch.tensor([r["bonds"] for r in results], dtype=torch.float32, device=device)
    mw_arr = torch.tensor([r["mw"] for r in results], dtype=torch.float32, device=device)

    # Correlations
    corr_mt = torch.corrcoef(torch.stack([ma_arr, target_arr]))[0, 1].item()
    corr_bt = torch.corrcoef(torch.stack([bonds_arr, target_arr]))[0, 1].item()
    corr_wt = torch.corrcoef(torch.stack([mw_arr, target_arr]))[0, 1].item()
    corr_mb = torch.corrcoef(torch.stack([ma_arr, bonds_arr]))[0, 1].item()

    # MA/Bonds ratio correlation with target
    ratio = ma_arr / torch.clamp(bonds_arr, min=1)
    corr_rt = torch.corrcoef(torch.stack([ratio, target_arr]))[0, 1].item()

    log(f"\n  Statistics:")
    log(f"    MA: mean={ma_arr.mean().item():.2f}, std={ma_arr.std().item():.2f}, range={int(ma_arr.min())}-{int(ma_arr.max())}")
    log(f"    {target_name}: mean={target_arr.mean().item():.2f}, std={target_arr.std().item():.2f}")

    log(f"\n  Correlations with {target_name}:")
    log(f"    MA vs {target_name}:        r = {corr_mt:.4f}")
    log(f"    Bonds vs {target_name}:     r = {corr_bt:.4f}")
    log(f"    MW vs {target_name}:        r = {corr_wt:.4f}")
    log(f"    MA/Bonds vs {target_name}:  r = {corr_rt:.4f}")

    # Is MA BETTER than bonds or MW at predicting the target?
    ma_better_bonds = abs(corr_mt) > abs(corr_bt)
    ma_better_mw = abs(corr_mt) > abs(corr_wt)
    ratio_better = abs(corr_rt) > abs(corr_mt)

    log(f"\n  MA better predictor than Bonds? {'YES' if ma_better_bonds else 'NO'} ({abs(corr_mt):.4f} vs {abs(corr_bt):.4f})")
    log(f"  MA better predictor than MW?    {'YES' if ma_better_mw else 'NO'} ({abs(corr_mt):.4f} vs {abs(corr_wt):.4f})")
    log(f"  MA/Bonds ratio better than MA?  {'YES' if ratio_better else 'NO'} ({abs(corr_rt):.4f} vs {abs(corr_mt):.4f})")

    # Bin analysis: MA bins vs mean target
    log(f"\n  {target_name} by MA bin:")
    log(f"    {'MA bin':>10} {'Count':>6} {'Mean '+target_name:>15}")
    for ma_lo in range(0, int(ma_arr.max().item()) + 1, 3):
        mask = (ma_arr >= ma_lo) & (ma_arr < ma_lo + 3)
        count = mask.sum().item()
        if count < 5: continue
        mean_t = target_arr[mask].mean().item()
        log(f"    {ma_lo:>3}-{ma_lo+2:>3} {count:>6} {mean_t:>15.3f}")

    return {
        "name": name, "n": len(results),
        "corr_ma_target": corr_mt, "corr_bonds_target": corr_bt,
        "corr_mw_target": corr_wt, "corr_ratio_target": corr_rt,
    }

# === Main ===
log("=" * 60)
log("REAL-WORLD DRUG DATA ANALYSIS")
log("Assembly Index vs Drug Properties")
log("=" * 60)

datasets = [
    ("ESOL (Solubility)",
     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
     "smiles", "measured log solubility in mols per litre", "LogS"),

    ("Lipophilicity",
     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
     "smiles", "exp", "LogD"),

    ("BBBP (Blood-Brain Barrier)",
     "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
     "smiles", "p_np", "BBB_penetration"),
]

all_dataset_results = []
for name, url, smi_col, tgt_col, tgt_name in datasets:
    result = process_dataset(name, url, smi_col, tgt_col, tgt_name)
    if result:
        all_dataset_results.append(result)

# Summary
log(f"\n{'='*60}")
log("GRAND SUMMARY: Does Assembly Index predict drug properties?")
log(f"{'='*60}")

log(f"\n{'Dataset':<30} {'N':>6} {'|r(MA)|':>8} {'|r(Bond)|':>9} {'|r(MW)|':>8} {'MA wins?':>9}")
log("-" * 75)
ma_wins = 0
for r in all_dataset_results:
    ma_r = abs(r["corr_ma_target"])
    bond_r = abs(r["corr_bonds_target"])
    mw_r = abs(r["corr_mw_target"])
    wins = ma_r > bond_r and ma_r > mw_r
    if wins: ma_wins += 1
    log(f"{r['name']:<30} {r['n']:>6} {ma_r:>8.4f} {bond_r:>9.4f} {mw_r:>8.4f} {'YES' if wins else 'no':>9}")

log(f"\nMA wins in {ma_wins}/{len(all_dataset_results)} datasets")

if ma_wins > 0:
    log(f"\nDISCOVERY: Assembly Index is a BETTER predictor of drug properties")
    log(f"than simple bond count or molecular weight in {ma_wins} dataset(s)!")
    log(f"This means molecular COMPLEXITY (not just size) matters for drug behavior.")
else:
    log(f"\nMA does not beat simple descriptors. But MA/Bonds ratio might add value")
    log(f"as a complementary feature in ML models.")

outfile.close()
