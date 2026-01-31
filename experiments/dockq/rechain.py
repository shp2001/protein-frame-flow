#!/usr/bin/env python3

from pathlib import Path
import string

root = Path(
    "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.4.0_loop_ppi/"
    "2026-01-29_00-29-37/epoch=57-step=41470/benchmark_wo_perturb_w_ag_cond"
)

new_chain_ids = list(string.ascii_uppercase)

def renumber_chains_in_pdb(pdb_path: Path):

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    chain_map = {}
    next_idx = 0
    new_lines = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM", "TER")):
            # PDB chain column (22, 1-based)
            old_chain = line[21]

            if old_chain not in chain_map:
                if next_idx >= len(new_chain_ids):
                    raise RuntimeError(
                        f"Too many chains in {pdb_path}"
                    )
                chain_map[old_chain] = new_chain_ids[next_idx]
                next_idx += 1

            new_chain = chain_map[old_chain]
            line = line[:21] + new_chain + line[22:]

        new_lines.append(line)

    with open(pdb_path, "w") as f:
        f.writelines(new_lines)

    return chain_map


if __name__ == "__main__":

    pdb_files = list(root.rglob("*.pdb"))

    for pdb in pdb_files:
        chain_map = renumber_chains_in_pdb(pdb)
        print(pdb, chain_map)
