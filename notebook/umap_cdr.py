import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from multiprocessing import Pool
from tqdm import tqdm

# -------------------------------------------------------
# paths
# -------------------------------------------------------
affinity_csv = "/home/psh/BioBetter/ai-assisted-affinity-maturation/merged_affinity_data.csv"

root_dir = (
    "/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.4.0_loop_ppi/2026-01-29_00-29-37/epoch=57-step=41470/5ggs_mean_pooling"
)

num_cpu = 20
save_dir = "/home/psh/protein-frame-flow/notebook/cdr_umap/CDRFlow_v2.4.0_loop_ppi/"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------------------------
# load affinity table
# -------------------------------------------------------
df = pd.read_csv(affinity_csv)

df["key"] = df["PDB"].astype(str) + "_" + df["Variant"].astype(str)

cdr_dict = dict(zip(df["key"], df["CDR"]))

# -------------------------------------------------------
# worker
# -------------------------------------------------------
def load_one_dir(args):
    name, root_dir, cdr_dict = args

    subdir = os.path.join(root_dir, name)

    if not os.path.isdir(subdir):
        return None

    if "_" not in name:
        return None

    pdb, variant = name.split("_", 1)
    key = f"{pdb}_{variant}"

    if key not in cdr_dict:
        return None

    pt_files = [f for f in os.listdir(subdir) if f.endswith(".pt")]
    if len(pt_files) != 1:
        print(f"skip (pt file != 1): {subdir}")
        return None

    pt_path = os.path.join(subdir, pt_files[0])

    vec = torch.load(pt_path, map_location="cpu")
    vec = vec.reshape(-1).numpy()

    cdr = cdr_dict[key]

    return vec, cdr, key, pdb


# -------------------------------------------------------
# collect embeddings (multi cpu)
# -------------------------------------------------------
names_in_dir = sorted(os.listdir(root_dir))
args = [(name, root_dir, cdr_dict) for name in names_in_dir]

results = []

with Pool(processes=num_cpu) as pool:
    for r in tqdm(
        pool.imap_unordered(load_one_dir, args),
        total=len(args),
        desc="Loading .pt files"
    ):
        results.append(r)

embeddings = []
cdr_values = []
names = []

pdb_for_save = None

for r in results:
    if r is None:
        continue

    vec, cdr, key, pdb = r

    embeddings.append(vec)
    cdr_values.append(cdr)
    names.append(key)

    if pdb_for_save is None:
        pdb_for_save = pdb

embeddings = np.stack(embeddings, axis=0)
cdr_values = np.array(cdr_values)

print("Total samples:", embeddings.shape[0])

# -------------------------------------------------------
# UMAP
# -------------------------------------------------------
reducer = umap.UMAP(
    n_components=2,
    random_state=42
)

emb_2d = reducer.fit_transform(embeddings)

# -------------------------------------------------------
# CDR category & colors
# -------------------------------------------------------
categories = cdr_values

unique_cdrs = sorted(pd.unique(categories))

cmap = plt.get_cmap("tab10")
color_map = {
    cdr: cmap(i % 10) for i, cdr in enumerate(unique_cdrs)
}

# -------------------------------------------------------
# plot
# -------------------------------------------------------
plt.figure(figsize=(8, 7))

for cat in unique_cdrs:
    mask = categories == cat
    if mask.sum() == 0:
        continue

    plt.scatter(
        emb_2d[mask, 0],
        emb_2d[mask, 1],
        s=40,
        label=str(cat),
        c=[color_map[cat]],
        alpha=0.8
    )

plt.legend(title="CDR")
plt.title(f"{pdb_for_save}")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()

plt.savefig(os.path.join(save_dir, f"{pdb_for_save}.png"))
plt.close()
