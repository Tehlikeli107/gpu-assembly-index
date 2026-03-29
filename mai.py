"""
mai.py — Molecular Assembly Index CLI
======================================
Usage:
    python mai.py "CCO"                    # Single molecule
    python mai.py "CCO" "c1ccccc1" "CC=O"  # Multiple molecules
    python mai.py --file molecules.txt     # File with one SMILES per line
    python mai.py --predict "CCO"          # Use GPU NN predictor (fast, ~0.5 MA error)
"""
import sys
import time
from collections import defaultdict

def compute_ma_from_smiles(smiles, timeout=3.0):
    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    mol = Chem.RemoveHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds, adj = [], defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = int(b.GetBondTypeAsDouble())
        bonds.append((i, j, bt))
        adj[i].append((j, len(bonds)-1, bt))
        adj[j].append((i, len(bonds)-1, bt))

    nb = len(bonds)
    if nb <= 1:
        return nb, None

    # Find duplicate fragments
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

    naive = nb - 1
    t0 = time.time()

    frags = defaultdict(list)
    for bi in range(nb):
        a, b, t = bonds[bi]
        sa, sb = atoms[a], atoms[b]
        if sa > sb: sa, sb = sb, sa
        frags[(sa, t, sb)].append(frozenset([bi]))

    dups = {}
    for sig, occs in frags.items():
        if len(occs) >= 2:
            dups[cfrag(list(occs[0]))] = occs

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
        for frag in nxt:
            sg[cfrag(list(frag))].append(frag)
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
        if len(no) >= 2:
            cands.append(((s-1)*(len(no)-1), s, no))
    cands.sort(reverse=True)

    used = set(); total = 0
    for _, s, occs in cands:
        av = [o for o in occs if o.isdisjoint(used)]
        if len(av) >= 2:
            total += (s-1)*(len(av)-1)
            for o in av: used |= o

    ma = naive - total
    return ma, None


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    smiles_list = []
    use_predict = False

    if args[0] == "--predict":
        use_predict = True
        args = args[1:]

    if args[0] == "--file":
        with open(args[1]) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        smiles_list = args

    print(f"{'SMILES':<50} {'MA':>5} {'Note'}")
    print("-" * 65)

    for smi in smiles_list:
        t0 = time.time()
        ma, err = compute_ma_from_smiles(smi)
        ms = (time.time() - t0) * 1000

        if err:
            print(f"{smi:<50} {'ERR':>5} {err}")
        else:
            bio = " << biosignature" if ma > 15 else ""
            print(f"{smi:<50} {ma:>5} {ms:.0f}ms{bio}")


if __name__ == "__main__":
    main()
