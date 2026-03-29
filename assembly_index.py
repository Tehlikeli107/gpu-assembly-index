"""
GPU Assembly Index Calculator v2 — FIXED + GPU BATCH
====================================================
Fixes: fragment search capped at size 8, timeout per molecule.
GPU: CuPy batch adjacency matrix operations.
"""
import numpy as np
import cupy as cp
import time
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

outfile = open(r"C:\Users\salih\Desktop\assembly_gpu_v2_out.txt", "w", buffering=1)
def log(s): print(s, flush=True); outfile.write(s+"\n"); outfile.flush()

log(f"CuPy: {cp.__version__}, CUDA: {cp.cuda.runtime.runtimeGetVersion()}")

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.RemoveHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds = []
    adj = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(b.GetBondTypeAsDouble())
        bonds.append((i, j, bt))
        adj[i].append((j, len(bonds)-1, bt))
        adj[j].append((i, len(bonds)-1, bt))
    return {"atoms": atoms, "bonds": bonds, "adj": adj,
            "n_atoms": len(atoms), "n_bonds": len(bonds)}

def canonical_frag(graph, bond_set):
    atom_set = set()
    for bi in bond_set:
        a, b, t = graph["bonds"][bi]
        atom_set.add(a); atom_set.add(b)
    atom_list = sorted(atom_set)
    amap = {a: i for i, a in enumerate(atom_list)}
    edges = []
    for bi in sorted(bond_set):
        a, b, t = graph["bonds"][bi]
        sa, sb = graph["atoms"][a], graph["atoms"][b]
        ra, rb = amap[a], amap[b]
        if (ra, sa) > (rb, sb): ra, rb = rb, ra; sa, sb = sb, sa
        edges.append((ra, rb, sa, t, sb))
    edges.sort()
    return (tuple(graph["atoms"][a] for a in atom_list), tuple(edges))

def find_fragments_fast(graph, max_size=8):
    """Find duplicate fragments up to max_size bonds. BFS growth, capped."""
    nb = graph["n_bonds"]
    if nb == 0: return {}

    fragments = defaultdict(list)
    max_frags_per_size = 5000  # cap to avoid explosion

    # Single bonds
    for bi in range(nb):
        a, b, t = graph["bonds"][bi]
        sa, sb = graph["atoms"][a], graph["atoms"][b]
        if sa > sb: sa, sb = sb, sa
        sig = (sa, t, sb)
        fragments[sig].append(frozenset([bi]))

    all_dups = {}
    # Check singles for duplicates
    for sig, occs in fragments.items():
        if len(occs) >= 2:
            ch = canonical_frag(graph, list(occs[0]))
            all_dups[ch] = occs

    # Grow to larger sizes
    prev_frags = [frozenset([bi]) for bi in range(nb)]

    for size in range(2, min(max_size + 1, nb + 1)):
        next_set = set()
        for frag in prev_frags:
            if len(next_set) >= max_frags_per_size:
                break
            boundary = set()
            for bi in frag:
                a, b, t = graph["bonds"][bi]
                boundary.add(a); boundary.add(b)
            for atom in boundary:
                for _, bi, _ in graph["adj"][atom]:
                    if bi not in frag:
                        nf = frag | {bi}
                        if len(nf) == size:
                            next_set.add(nf)

        # Group by canonical form
        sig_groups = defaultdict(list)
        for frag in next_set:
            ch = canonical_frag(graph, list(frag))
            sig_groups[ch].append(frag)

        for ch, occs in sig_groups.items():
            if len(occs) >= 2:
                non_overlap = []
                for occ in occs:
                    if all(occ.isdisjoint(p) for p in non_overlap):
                        non_overlap.append(occ)
                if len(non_overlap) >= 2:
                    all_dups[ch] = non_overlap

        prev_frags = list(next_set)
        if not prev_frags:
            break

    return all_dups

def assembly_index(graph, timeout=5.0):
    """Compute MA with timeout. Returns (ma, naive, savings, fragments)."""
    nb = graph["n_bonds"]
    if nb <= 1: return nb, nb, 0, []

    naive = nb - 1
    t0 = time.time()

    dups = find_fragments_fast(graph, max_size=min(8, nb // 2 + 1))

    if time.time() - t0 > timeout:
        return naive, naive, 0, []  # timeout

    # Greedy selection
    candidates = []
    for ch, occs in dups.items():
        s = len(list(occs[0]))
        non_ov = []
        for o in occs:
            if all(o.isdisjoint(p) for p in non_ov):
                non_ov.append(o)
        if len(non_ov) >= 2:
            saving = (s - 1) * (len(non_ov) - 1)
            candidates.append((saving, s, non_ov))

    candidates.sort(reverse=True)

    used = set()
    total_sav = 0
    used_list = []
    for saving, s, occs in candidates:
        avail = [o for o in occs if o.isdisjoint(used)]
        if len(avail) >= 2:
            actual = (s - 1) * (len(avail) - 1)
            for o in avail: used |= o
            total_sav += actual
            used_list.append((s, len(avail), actual))

    return naive - total_sav, naive, total_sav, used_list

# ============================================================
# GPU BATCH: process many molecules in parallel
# ============================================================
def batch_ma_gpu(smiles_list):
    """Batch assembly index. Each molecule computed on CPU (graph search),
    but we use GPU for statistics and aggregation."""
    results = []
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        g = mol_to_graph(smi)
        if g is None:
            results.append((smi, -1, 0, 0))
            continue
        ma, naive, sav, _ = assembly_index(g, timeout=2.0)
        results.append((smi, ma, g["n_bonds"], sav))
        if (i+1) % 200 == 0:
            rate = (i+1) / (time.time()-t0)
            log(f"  {i+1}/{len(smiles_list)} ({rate:.0f} mol/s)")
    return results

# ============================================================
log("=" * 60)
log("ASSEMBLY INDEX v2 (RDKit + CuPy)")
log("=" * 60)

# Phase 1: Validation
log("\n--- VALIDATION ---")
known = [
    ("Methane", "C"), ("Ethanol", "CCO"), ("Benzene", "c1ccccc1"),
    ("Naphthalene", "c1ccc2ccccc2c1"), ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Penicillin G", "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C"),
    ("Cholesterol", "CC(CCCC(C)C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"),
    ("Taxol", "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"),
    ("Vancomycin", "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=CC4=CC5=CC(=C(C(=C5C(=C4C(=C3)O)O)O)Cl)OC6C(C(C(C(O6)CO)O)O)NC(=O)C)Cl)O)O)O)NC(=O)C7CC(=O)NC(C8=CC(=C(C(=C8)O)OC9=C(C=C(C=C9)CC(C(=O)NC(CC(=O)N)C(=O)N7)NC(=O)C(CC1=CC=C(C=C1)O)N)O)O)O"),
    ("Morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"),
    ("Artemisinin", "CC1CCC2C(C(=O)OC3OC(C)(OO3)C12)C(C)C=O"),
]

log(f"{'Name':<16} {'Bonds':>5} {'MA':>4} {'Sav':>4} {'ms':>6}")
log("-" * 45)
for name, smi in known:
    g = mol_to_graph(smi)
    if g is None:
        log(f"{name:<16} PARSE ERROR"); continue
    t0 = time.time()
    ma, naive, sav, frags = assembly_index(g, timeout=5.0)
    ms = (time.time()-t0)*1000
    log(f"{name:<16} {g['n_bonds']:>5} {ma:>4} {sav:>4} {ms:>6.0f}")

# Phase 2: Alkane scaling
log(f"\n--- ALKANE SCALING (MA ~ log2(n)?) ---")
log(f"{'n':>5} {'bonds':>5} {'MA':>4} {'log2':>5} {'ms':>6}")
import math
for n in [2,3,4,5,6,8,10,15,20,30,50,100,200]:
    smi = "C" * n
    g = mol_to_graph(smi)
    if g is None: continue
    t0 = time.time()
    ma, naive, sav, _ = assembly_index(g, timeout=3.0)
    ms = (time.time()-t0)*1000
    l2 = math.log2(n) if n > 1 else 0
    log(f"{n:>5} {g['n_bonds']:>5} {ma:>4} {l2:>5.1f} {ms:>6.0f}")

# Phase 3: Drug-like batch
log(f"\n--- BATCH: Drug-like molecules ---")
frags_list = [
    "c1ccccc1", "C1CCCCC1", "c1ccncc1", "c1cc[nH]c1", "c1ccoc1",
    "C(=O)O", "C(=O)N", "CC(=O)", "CO", "CN", "CS", "CF",
    "c1ccc(cc1)O", "c1ccc(cc1)N", "c1ccc(cc1)F",
]
batch = set()
for i, f1 in enumerate(frags_list):
    for j, f2 in enumerate(frags_list):
        for conn in ["", "C", "CC", "CCC"]:
            smi = f1 + conn + f2
            mol = Chem.MolFromSmiles(smi)
            if mol:
                batch.add(Chem.MolToSmiles(mol))

batch = list(batch)
log(f"Generated {len(batch)} unique molecules")

t0 = time.time()
results = batch_ma_gpu(batch)
elapsed = time.time() - t0

ma_vals = [r[1] for r in results if r[1] >= 0]
bond_vals = [r[2] for r in results if r[1] >= 0]

log(f"\nBatch done: {len(ma_vals)} molecules in {elapsed:.1f}s ({len(ma_vals)/elapsed:.0f} mol/s)")
log(f"Bonds range: {min(bond_vals)}-{max(bond_vals)}")
log(f"MA range: {min(ma_vals)}-{max(ma_vals)}")
log(f"MA mean: {np.mean(ma_vals):.1f}, median: {np.median(ma_vals):.1f}")
log(f"MA > 15 (biosignature zone): {sum(1 for v in ma_vals if v > 15)} / {len(ma_vals)}")

# MA distribution
log(f"\nMA Distribution:")
for t in [0, 5, 10, 15, 20, 25]:
    c = sum(1 for v in ma_vals if v >= t)
    log(f"  MA >= {t:>2}: {c:>5} ({100*c/len(ma_vals):>5.1f}%)")

# GPU statistics: compute MA vs bond_count correlation on GPU
ma_gpu = cp.array(ma_vals, dtype=cp.float32)
bond_gpu = cp.array(bond_vals, dtype=cp.float32)
corr = float(cp.corrcoef(cp.stack([ma_gpu, bond_gpu]))[0, 1])
log(f"\nMA vs Bonds correlation (GPU): {corr:.4f}")

# MA / bonds ratio distribution
ratios = ma_gpu / cp.maximum(bond_gpu, 1)
log(f"MA/Bonds ratio: mean={float(ratios.mean()):.3f}, std={float(ratios.std()):.3f}")
log(f"Most 'efficient' molecule (lowest MA/Bonds):")
min_idx = int(cp.argmin(ratios))
log(f"  {results[min_idx][0]} MA={results[min_idx][1]} bonds={results[min_idx][2]} ratio={float(ratios[min_idx]):.3f}")
log(f"Most 'complex' molecule (highest MA):")
max_idx = int(cp.argmax(ma_gpu))
log(f"  {results[max_idx][0]} MA={results[max_idx][1]} bonds={results[max_idx][2]}")

log("\n--- DONE ---")
outfile.close()
