"""
PubChem Bulk Scan — Download + GPU Assembly Index
==================================================
Download PubChem compound SMILES via listkey API, compute MA at scale.
Strategy: use PubChem's classification lists (approved drugs, natural products, etc.)
"""
import numpy as np
import cupy as cp
import time
import urllib.request
import json
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

outfile = open(r"C:\Users\salih\Desktop\pubchem_scan_out.txt", "w", buffering=1)
def log(s): print(s, flush=True); outfile.write(s+"\n"); outfile.flush()

# === Compact MA computation ===
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

# === PubChem download via classification ===
def get_pubchem_by_source(source_name, max_cids=5000):
    """Get CIDs from PubChem classification tree."""
    # Use sdq (structure-data query) for specific sources
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{source_name}/property/IsomericSMILES/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AssemblyGPU/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        props = data.get("PropertyTable", {}).get("Properties", [])
        return [(p.get("CID", 0), p.get("IsomericSMILES") or p.get("SMILES", "")) for p in props if p.get("IsomericSMILES") or p.get("SMILES")]
    except:
        return []

def get_pubchem_cids(start, count):
    """Get molecules by CID range, skipping invalid."""
    cids = list(range(start, start + count))
    # Try smaller batches
    results = []
    for chunk_start in range(0, len(cids), 50):
        chunk = cids[chunk_start:chunk_start+50]
        cids_str = ",".join(str(c) for c in chunk)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/IsomericSMILES/JSON"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AssemblyGPU/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            props = data.get("PropertyTable", {}).get("Properties", [])
            for p in props:
                smi = p.get("IsomericSMILES") or p.get("SMILES")
                if smi:
                    results.append((p["CID"], smi))
        except:
            pass
        time.sleep(0.2)  # rate limit
    return results

log("=" * 60)
log("PUBCHEM BULK ASSEMBLY INDEX SCAN")
log("=" * 60)

# Download molecules from PubChem in chunks
all_results = []
total_t0 = time.time()

# Scan CID ranges that are known to have compounds
log("\nPhase 1: Scanning PubChem CID ranges...")
cid_ranges = [
    (2000, 500), (3000, 500), (4000, 500), (5000, 500),
    (6000, 500), (7000, 500), (8000, 500), (10000, 500),
    (15000, 500), (20000, 500), (50000, 500), (100000, 500),
]

for start, count in cid_ranges:
    t0 = time.time()
    mols = get_pubchem_cids(start, count)
    dl_time = time.time() - t0

    if not mols:
        log(f"  CID {start}-{start+count}: no data")
        continue

    batch_ma = []
    for cid, smi in mols:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        ma = compute_ma(mol, timeout=1.5)
        if ma >= 0:
            nb = mol.GetNumBonds()
            batch_ma.append((cid, smi, ma, nb))
            all_results.append((cid, smi, ma, nb))

    if batch_ma:
        mas = [m[2] for m in batch_ma]
        log(f"  CID {start}-{start+count}: {len(batch_ma)} molecules, "
            f"MA range {min(mas)}-{max(mas)}, mean={np.mean(mas):.1f} "
            f"({dl_time:.0f}s dl, {len(batch_ma)/(time.time()-t0-dl_time+0.01):.0f} mol/s)")

elapsed = time.time() - total_t0
log(f"\nTotal: {len(all_results)} molecules in {elapsed:.0f}s")

if all_results:
    # GPU statistics
    ma_arr = cp.array([r[2] for r in all_results], dtype=cp.float32)
    bonds_arr = cp.array([r[3] for r in all_results], dtype=cp.float32)

    log(f"\nMA Statistics (GPU):")
    log(f"  Mean: {float(ma_arr.mean()):.2f}")
    log(f"  Std: {float(ma_arr.std()):.2f}")
    log(f"  Min: {int(ma_arr.min())}, Max: {int(ma_arr.max())}")
    log(f"  Median: {float(cp.median(ma_arr)):.1f}")
    log(f"  MA vs Bonds corr: {float(cp.corrcoef(cp.stack([ma_arr, bonds_arr]))[0,1]):.4f}")

    log(f"\nMA Distribution:")
    for t in [0, 5, 10, 15, 20, 30, 50]:
        c = int((ma_arr >= t).sum())
        if c > 0:
            log(f"  MA >= {t:>2}: {c:>5} ({100*c/len(all_results):>5.1f}%)")

    log(f"\nBiosignature (MA > 15):")
    bio = [(cid, smi, ma, nb) for cid, smi, ma, nb in all_results if ma > 15]
    log(f"  {len(bio)} / {len(all_results)} ({100*len(bio)/len(all_results):.1f}%)")

    log(f"\nTop 30 highest MA from PubChem:")
    all_results.sort(key=lambda x: -x[2])
    for cid, smi, ma, nb in all_results[:30]:
        log(f"  CID={cid:>8} MA={ma:>3} bonds={nb:>3} {smi[:55]}")

log(f"\n--- DONE ---")
outfile.close()
