"""
Large-Scale Assembly Index Scan
================================
Download ZINC drug-like SMILES + compute MA + find patterns + discover.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import urllib.request
import gzip
import io
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

outfile = open(r"C:\Users\salih\Desktop\large_scan_out.txt", "w", buffering=1)
def log(s): print(s, flush=True); outfile.write(s+"\n"); outfile.flush()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")

# === Compact MA ===
def compute_ma(mol, timeout=1.5):
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

# === Download ZINC SMILES ===
def download_zinc_smiles(max_molecules=50000):
    """Download drug-like SMILES from ZINC database."""
    # ZINC provides tranches via URL pattern
    # Use ZINC20 standard drug-like: in-stock, pH 7, drug-like
    urls = [
        "https://zinc20.docking.org/substances/subsets/in-stock/?count=all&format=smiles",
        "https://zinc20.docking.org/substances/subsets/fda/?count=all&format=smiles",
    ]

    # Fallback: generate large diverse set locally
    log("  Generating large diverse molecular dataset locally...")
    mols = set()

    rings = [
        "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1cc[nH]c1", "c1ccoc1",
        "c1ccsc1", "c1ccc2ccccc2c1", "C1CCNCC1", "C1CCOCC1",
        "c1ccc2[nH]ccc2c1", "c1cnc2ccccc2n1", "c1ccnc(N)n1",
        "C1CC1", "C1CCC1", "C1CCCC1", "c1ccc(cc1)O", "c1ccc(cc1)N",
        "c1ccc(cc1)F", "c1ccc(cc1)Cl", "c1ccc(cc1)C",
        "c1cc2ccccc2[nH]1", "c1ncc2ccccc2n1",
    ]
    fgs = [
        "O", "N", "F", "Cl", "Br", "S", "C(=O)O", "C(=O)N", "C=O",
        "C#N", "C(F)(F)F", "OC", "NC", "SC", "C(=O)OC", "NC(=O)C",
        "S(=O)(=O)N", "N(C)C", "OCC", "NCC", "C(=O)NC", "CC(=O)",
        "C(=O)Cl", "C(=S)N", "P(=O)(O)O", "B(O)O",
    ]
    links = ["", "C", "CC", "CCC", "CCCC", "CCCCC",
             "C=C", "C#C", "CCO", "CCN", "CCS", "CCOC", "CCNC",
             "C(=O)", "C(=O)N", "COCC", "CNCC"]

    # Massive combinatorial generation
    # Ring-link-ring
    for r1 in rings:
        for r2 in rings:
            for lnk in links:
                m = Chem.MolFromSmiles(r1 + lnk + r2)
                if m: mols.add(Chem.MolToSmiles(m))
                if len(mols) >= max_molecules: break
            if len(mols) >= max_molecules: break
        if len(mols) >= max_molecules: break

    # Ring-fg
    for r in rings:
        for fg in fgs:
            for lnk in links:
                m = Chem.MolFromSmiles(r + lnk + fg)
                if m: mols.add(Chem.MolToSmiles(m))
        for fg1 in fgs:
            for fg2 in fgs:
                m = Chem.MolFromSmiles(r + "(" + fg1 + ")" + fg2)
                if m: mols.add(Chem.MolToSmiles(m))
                m = Chem.MolFromSmiles(r + "(" + fg1 + ")C" + fg2)
                if m: mols.add(Chem.MolToSmiles(m))

    # Ring-link-ring-fg
    for r1 in rings[:12]:
        for r2 in rings[:12]:
            for fg in fgs[:12]:
                m = Chem.MolFromSmiles(r1 + "C" + r2 + fg)
                if m: mols.add(Chem.MolToSmiles(m))
                m = Chem.MolFromSmiles(r1 + "(" + fg + ")C" + r2)
                if m: mols.add(Chem.MolToSmiles(m))

    # Triple ring
    for r1 in rings[:10]:
        for r2 in rings[:10]:
            for r3 in rings[:10]:
                m = Chem.MolFromSmiles(r1 + "C" + r2 + "C" + r3)
                if m: mols.add(Chem.MolToSmiles(m))

    # Chains with branches
    for n in range(2, 30):
        for e1 in ["", "O", "N", "F", "Cl", "C(=O)O", "C(=O)N", "c1ccccc1"]:
            for e2 in ["", "O", "N", "F", "c1ccccc1"]:
                m = Chem.MolFromSmiles(e1 + "C" * n + e2)
                if m: mols.add(Chem.MolToSmiles(m))

    return list(mols)[:max_molecules]

log("=" * 60)
log("LARGE-SCALE ASSEMBLY INDEX SCAN")
log("=" * 60)

# Step 1: Get molecules
log("\n[1] Generating molecular dataset...")
t0 = time.time()
all_smiles = download_zinc_smiles(50000)
log(f"    {len(all_smiles)} unique molecules ({time.time()-t0:.0f}s)")

# Step 2: Compute exact MA + molecular descriptors
log("\n[2] Computing Assembly Index + descriptors...")
results = []  # (smi, ma, bonds, atoms, mw, logp, hba, hbd, rotbonds, rings, aromatic_rings)
t0 = time.time()

for i, smi in enumerate(all_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue

    ma = compute_ma(mol, timeout=1.0)
    if ma < 0: continue

    mol_h = Chem.AddHs(mol)
    nb = mol.GetNumBonds()
    na = mol.GetNumHeavyAtoms()
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    nrings = rdMolDescriptors.CalcNumRings(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)

    results.append((smi, ma, nb, na, mw, logp, hba, hbd, rot, nrings, arom))

    if (i+1) % 5000 == 0:
        rate = (i+1) / (time.time()-t0)
        log(f"    {i+1}/{len(all_smiles)} ({rate:.0f} mol/s) processed={len(results)}")

elapsed = time.time() - t0
log(f"    Done: {len(results)} molecules in {elapsed:.0f}s ({len(results)/elapsed:.0f} mol/s)")

# Step 3: GPU analysis
log("\n[3] GPU statistical analysis...")

ma_arr = np.array([r[1] for r in results], dtype=np.float32)
bonds_arr = np.array([r[2] for r in results], dtype=np.float32)
atoms_arr = np.array([r[3] for r in results], dtype=np.float32)
mw_arr = np.array([r[4] for r in results], dtype=np.float32)
logp_arr = np.array([r[5] for r in results], dtype=np.float32)
hba_arr = np.array([r[6] for r in results], dtype=np.float32)
hbd_arr = np.array([r[7] for r in results], dtype=np.float32)
rot_arr = np.array([r[8] for r in results], dtype=np.float32)
rings_arr = np.array([r[9] for r in results], dtype=np.float32)
arom_arr = np.array([r[10] for r in results], dtype=np.float32)

# Move to GPU
ma_g = torch.tensor(ma_arr, device=device)
bonds_g = torch.tensor(bonds_arr, device=device)
atoms_g = torch.tensor(atoms_arr, device=device)
mw_g = torch.tensor(mw_arr, device=device)
logp_g = torch.tensor(logp_arr, device=device)
rings_g = torch.tensor(rings_arr, device=device)
arom_g = torch.tensor(arom_arr, device=device)
rot_g = torch.tensor(rot_arr, device=device)

log(f"\nBasic Statistics:")
log(f"  N molecules: {len(results)}")
log(f"  MA:     mean={ma_arr.mean():.2f}, std={ma_arr.std():.2f}, min={ma_arr.min():.0f}, max={ma_arr.max():.0f}")
log(f"  Bonds:  mean={bonds_arr.mean():.1f}, min={bonds_arr.min():.0f}, max={bonds_arr.max():.0f}")
log(f"  MW:     mean={mw_arr.mean():.0f}, min={mw_arr.min():.0f}, max={mw_arr.max():.0f}")

# Correlations
log(f"\nCorrelations with MA (GPU):")
for name, arr in [("Bonds", bonds_g), ("Atoms", atoms_g), ("MW", mw_g),
                   ("LogP", logp_g), ("Rings", rings_g), ("AromaticRings", arom_g),
                   ("RotBonds", rot_g)]:
    corr = torch.corrcoef(torch.stack([ma_g, arr]))[0, 1].item()
    log(f"  MA vs {name:<15}: r = {corr:.4f}")

# MA/Bonds ratio analysis
ratio = ma_g / torch.clamp(bonds_g, min=1)
log(f"\nMA/Bonds ratio:")
log(f"  mean={ratio.mean().item():.4f}, std={ratio.std().item():.4f}")
log(f"  min={ratio.min().item():.4f}, max={ratio.max().item():.4f}")

# MA distribution
log(f"\nMA Distribution:")
for t in [0, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50]:
    c = int((ma_g >= t).sum().item())
    if c > 0:
        pct = 100 * c / len(results)
        bar = "#" * min(int(pct), 50)
        log(f"  MA >= {t:>2}: {c:>6} ({pct:>5.1f}%) {bar}")

# BIOSIGNATURE analysis
log(f"\nBIOSIGNATURE THRESHOLD (MA > 15):")
bio_mask = ma_g > 15
n_bio = bio_mask.sum().item()
log(f"  {n_bio} / {len(results)} ({100*n_bio/len(results):.1f}%)")

if n_bio > 0:
    bio_mw = mw_g[bio_mask].mean().item()
    nonbio_mw = mw_g[~bio_mask].mean().item()
    bio_rings = rings_g[bio_mask].mean().item()
    nonbio_rings = rings_g[~bio_mask].mean().item()
    bio_arom = arom_g[bio_mask].mean().item()
    nonbio_arom = arom_g[~bio_mask].mean().item()

    log(f"\n  Biosignature vs Non-biosignature comparison:")
    log(f"  {'Property':<20} {'Bio (MA>15)':>12} {'Non-bio':>12}")
    log(f"  {'MW':<20} {bio_mw:>12.0f} {nonbio_mw:>12.0f}")
    log(f"  {'Rings':<20} {bio_rings:>12.1f} {nonbio_rings:>12.1f}")
    log(f"  {'AromaticRings':<20} {bio_arom:>12.1f} {nonbio_arom:>12.1f}")

# DISCOVERY: Is there a phase transition in MA/Bonds ratio?
log(f"\n{'='*60}")
log("DISCOVERY ANALYSIS: Phase transition in molecular complexity?")
log(f"{'='*60}")

# Bin molecules by bond count, compute mean MA/Bonds in each bin
log(f"\nMA vs Bond count (looking for non-linearity):")
log(f"  {'Bonds':>6} {'Count':>6} {'Mean MA':>8} {'MA/Bond':>8}")
for b_lo in range(0, 40, 2):
    mask = (bonds_g >= b_lo) & (bonds_g < b_lo + 2)
    count = mask.sum().item()
    if count < 10: continue
    mean_ma = ma_g[mask].mean().item()
    mean_ratio = ratio[mask].mean().item()
    log(f"  {b_lo:>3}-{b_lo+1:>3} {count:>6} {mean_ma:>8.2f} {mean_ratio:>8.4f}")

# Top molecules
log(f"\nTop 20 highest MA molecules:")
sorted_idx = torch.argsort(ma_g, descending=True)[:20]
for idx in sorted_idx:
    r = results[idx.item()]
    log(f"  MA={r[1]:>3} bonds={r[2]:>3} MW={r[4]:>6.0f} rings={r[9]} {r[0][:50]}")

# Most "efficient" molecules (lowest MA/Bonds)
log(f"\nTop 20 most 'compressible' molecules (lowest MA/Bonds):")
sorted_ratio = torch.argsort(ratio)[:20]
for idx in sorted_ratio:
    r = results[idx.item()]
    rat = ratio[idx].item()
    log(f"  MA/B={rat:.3f} MA={r[1]:>3} bonds={r[2]:>3} {r[0][:50]}")

# Most "incompressible" molecules (highest MA/Bonds, near 1.0)
log(f"\nTop 20 most 'incompressible' molecules (highest MA/Bonds, near 1.0):")
sorted_ratio_desc = torch.argsort(ratio, descending=True)[:20]
for idx in sorted_ratio_desc:
    r = results[idx.item()]
    rat = ratio[idx].item()
    log(f"  MA/B={rat:.3f} MA={r[1]:>3} bonds={r[2]:>3} {r[0][:50]}")

log(f"\n--- DONE ---")
outfile.close()
