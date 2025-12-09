import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from Bio import PDB
from concurrent.futures import ProcessPoolExecutor, as_completed
import umap.umap_ as umap


# ---------------------------------------------
# 0. CDR residue ranges (PDB numbering)
# ---------------------------------------------
cdr_residues = {
    "h1": (23, 35),
    "h2": (50, 59),
    "h3": (97, 109)
}


def extract_mutation_index(mutation_str):
    parts = mutation_str.split("__")
    for p in parts:
        if p != "WT":
            m = re.search(r"[A-Z]+(\d+)[A-Z]+", p)
            if m:
                return int(m.group(1))
    return None


# ---------------------------------------------------
# 2. Build PDB residue mapping only once
# ---------------------------------------------------
def build_pdb_residue_map(example_pdb_path):
    print("Building residue → tensor index mapping using example PDB...")
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("example", example_pdb_path)

    residues_list = []
    for model in structure:
        for chain in model:
            chain_id = chain.id.upper()
            for res in chain:
                if PDB.is_aa(res):
                    res_id = res.get_id()[1]
                    residues_list.append((chain_id, res_id))

    print(f"Total residues in mapping: {len(residues_list)}")
    return residues_list


def chain_res_to_idx(chain_id, res_id, residues_list):
    chain_id = chain_id.upper()
    for i, (c, r) in enumerate(residues_list):
        if c == chain_id and r == res_id:
            return i
    return None


# ---------------------------------------------------
# 3. Processing each sample
# ---------------------------------------------------
def process_single_sample(args):
    folder, pairformer_dir, df, residues_list = args
    print(folder)
    full_path = os.path.join(pairformer_dir, folder)
    pt_path = os.path.join(full_path, "pair.pt")

    if not os.path.isfile(pt_path):
        return None

    data = torch.load(pt_path, map_location="cpu")
    z = data["z"]   # [L, L, C]
    L, _, C = z.shape

    # --------------------------------------------------------
    # 1) 체인 구분: antibody(H,L) vs antigen(last chain)
    # --------------------------------------------------------
    chain_ids = [c for (c, r) in residues_list]     # ex: ['H','H',...,'L','L',...,'A','A',...]
    unique_chain_order = []
    for c in chain_ids:
        if c not in unique_chain_order:
            unique_chain_order.append(c)

    # antibody = 첫 2개 체인
    ab_chains = unique_chain_order[:2]
    ag_chain  = unique_chain_order[-1]  # 마지막 체인

    ab_indices = [i for i,(c,r) in enumerate(residues_list) if c in ab_chains]
    ag_indices = [i for i,(c,r) in enumerate(residues_list) if c == ag_chain]

    if len(ab_indices)==0 or len(ag_indices)==0:
        return None

    # --------------------------------------------------------
    # 2) antibody–antigen interaction slice
    #    shape: [len(ab), len(ag), C]
    # --------------------------------------------------------
    ab_idx = torch.tensor(ab_indices)
    ag_idx = torch.tensor(ag_indices)

    # (Na, Ng, C)
    inter_1 = z[ab_idx][:, ag_idx, :]

    # (Ng, Na, C)
    inter_2_raw = z[ag_idx][:, ab_idx, :]

    # (Na, Ng, C) 로 transpose
    inter_2 = inter_2_raw.permute(1, 0, 2)

    # 양방향 평균
    inter = 0.5 * (inter_1 + inter_2)

    # 마지막 C 평균 → (Na, Ng)
    inter_mean = torch.mean(inter, axis=-1)

    # flatten → UMAP input
    feature = inter_mean.flatten().numpy()


    # --------------------------------------------------------
    # 4) label
    # --------------------------------------------------------
    if "WT__WT__WT" in folder:
        label = 1
    else:
        row = df[df["pdb_name"] == folder]
        if len(row) == 0:
            return None

        affinity_value = row["affinity"].values[0]
        label = 0 if affinity_value == -1 else 1
    return feature, label


# ---------------------------------------------------
# 4. Load metadata & residue mapping
# ---------------------------------------------------
pairformer_dir = "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_ori_perturb_stage2/2025-12-05_16-29-09/epoch=45-step=65918_copy/5ggs_optimab"
affinity_info = "/home/psh/BioBetter/5ggs/meta/metadata_all.csv"
example_pdb = "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_ori_perturb_stage3/2025-12-06_22-56-57/epoch=52-step=75949_copy/5ggs_dms/5ggs_H_L_A_Y35D__WT__WT/sample_0/sample_1.pdb"

df = pd.read_csv(affinity_info)
folders = os.listdir(pairformer_dir)

residues_list = build_pdb_residue_map(example_pdb)


# ---------------------------------------------------
# 5. Multiprocessing
# ---------------------------------------------------
features = []
labels = []

num_workers = 55
print(f"Using {num_workers} CPU workers...")

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [
        executor.submit(process_single_sample, (folder, pairformer_dir, df, residues_list))
        for folder in folders
    ]


    for f in as_completed(futures):
        res = f.result()
        if res is not None:
            feature, label = res
            features.append(feature)
            labels.append(label)

features = np.array(features)
labels = np.array(labels)

print("Loaded feature shape:", features.shape)


# ---------------------------------------------------
# 6. PCA
# ---------------------------------------------------
print("Running PCA reduction before UMAP...")
pca = PCA(n_components=50, random_state=42)
features_reduced = pca.fit_transform(features)
print("PCA reduced shape:", features_reduced.shape)


# ---------------------------------------------------
# 7. UMAP
# ---------------------------------------------------
print("Running UMAP...")
umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
)

embedding = umap_model.fit_transform(features_reduced)


# ---------------------------------------------------
# 8. Visualization
# ---------------------------------------------------
plt.figure(figsize=(7, 7))

color_map = {1: "#4682B4", 0: "#FF6347"}

for lab in [0, 1]:
    mask = (labels == lab)
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=color_map[lab],
        s=20,
        alpha=0.25,
        label="Binding" if lab == 1 else "Non-binding"
    )

plt.title("UMAP of Pairformer CDR-based Mutation Feature (PCA → UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.legend(title="Label")
plt.tight_layout()

save_path = "/home/psh/protein-frame-flow/notebook/umap_perturb_off_diagonal_stage_3.png"
plt.savefig(save_path, dpi=300)
print(f"Saved UMAP figure to: {save_path}")

plt.close()
