import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from multiprocessing import Pool
from tqdm import tqdm
import lzma
import math
import matplotlib as mpl
from sklearn.decomposition import PCA

# -------------------------------------------------------
# paths
# -------------------------------------------------------
affinity_csv = "/home/psh/BioBetter/ai-assisted-affinity-maturation/merged_affinity_data.csv"

root_dir = (
"/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.4.0_loop_ppi/2026-01-29_00-29-37/epoch=57-step=41470/3q1s_expand"
)

num_cpu = 20

save_dir = "/home/psh/protein-frame-flow/notebook/affinity_umap/CDRFlow_v2.4.0_loop_ppi/"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------------------------
# plotting option
# -------------------------------------------------------
POINT_ALPHA = 0.6

# -------------------------------------------------------
# load affinity table
# -------------------------------------------------------
df = pd.read_csv(affinity_csv)
df["key"] = df["PDB"].astype(str) + "_" + df["Variant"].astype(str)

kd_dict  = dict(zip(df["key"], df["KD (nM)"]))
cdr_dict = dict(zip(df["key"], df["CDR"]))

# -------------------------------------------------------
# loading helpers
# -------------------------------------------------------
def load_fp16_lzma(path, map_location="cpu"):
    with lzma.open(path, "rb") as f:
        x = torch.load(f, map_location=map_location)
    return x

# -------------------------------------------------------
# worker
# -------------------------------------------------------
def load_one_dir(args):
    name, root_dir, kd_dict = args

    subdir = os.path.join(root_dir, name)

    if not os.path.isdir(subdir):
        return None

    if "_" not in name:
        return None

    pdb, variant = name.split("_", 1)
    key = f"{pdb}_{variant}"

    if key not in kd_dict:
        return None

    files = [
        f for f in os.listdir(subdir)
        if f.endswith(".pt") or f.endswith(".xz")
    ]

    if len(files) != 1:
        print(f"skip (file != 1): {subdir}")
        return None

    path = os.path.join(subdir, files[0])

    if path.endswith(".pt"):
        vec = torch.load(path, map_location="cpu")
        is_xz = False
    elif path.endswith(".xz"):
        vec = load_fp16_lzma(path, map_location="cpu")
        is_xz = True
    else:
        return None

    vec = vec.reshape(-1).cpu().numpy()

    kd  = kd_dict[key]
    cdr = cdr_dict.get(key, "NA")

    return vec, kd, cdr, key, pdb, is_xz


# -------------------------------------------------------
# collect embeddings
# -------------------------------------------------------
names_in_dir = sorted(os.listdir(root_dir))
args = [(name, root_dir, kd_dict) for name in names_in_dir]

results = []

with Pool(processes=num_cpu) as pool:
    for r in tqdm(
        pool.imap_unordered(load_one_dir, args),
        total=len(args),
        desc="Loading embeddings"
    ):
        results.append(r)

embeddings = []
kd_values = []
cdr_values = []
is_xz_list = []

pdb_for_save = None

for r in results:
    if r is None:
        continue

    vec, kd, cdr, key, pdb, is_xz = r

    embeddings.append(vec)
    kd_values.append(kd)
    cdr_values.append(cdr)
    is_xz_list.append(is_xz)

    if pdb_for_save is None:
        pdb_for_save = pdb

embeddings = np.stack(embeddings, axis=0)
kd_values  = np.asarray(kd_values, dtype=float)
cdr_values = np.asarray(cdr_values)
is_xz_list = np.asarray(is_xz_list, dtype=bool)

print("Total samples:", embeddings.shape[0])

# -------------------------------------------------------
# UMAP per CDR and subplot
# -------------------------------------------------------
unique_cdrs = sorted(pd.unique(cdr_values))

n_cdr = len(unique_cdrs)
ncol = min(3, n_cdr)
nrow = math.ceil(n_cdr / ncol)

fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(10 * ncol, 8 * nrow),
    squeeze=False
)

for i, cdr in enumerate(unique_cdrs):

    ax = axes[i // ncol, i % ncol]

    mask = cdr_values == cdr

    if mask.sum() < 3:
        ax.set_title(f"{cdr} (too few)")
        ax.axis("off")
        continue

    emb_sub = embeddings[mask]
    kd_sub  = kd_values[mask]
    xz_sub  = is_xz_list[mask]

    # -----------------------------
    # PCA only if any xz in group
    # -----------------------------
    if np.any(xz_sub):
        n_comp = min(128, emb_sub.shape[1], emb_sub.shape[0] - 1)
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, random_state=42)
            emb_sub = pca.fit_transform(emb_sub)

    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    emb_2d = reducer.fit_transform(emb_sub)

    m_red    = kd_sub == -1
    m_blue   = (kd_sub >= 0) & (kd_sub < 20)
    m_purple = kd_sub >= 20

    # KD == -1
    if np.any(m_red):
        ax.scatter(
            emb_2d[m_red, 0],
            emb_2d[m_red, 1],
            s=30,
            c="red",
            alpha=POINT_ALPHA,
            label="KD = -1"
        )

    # 0 <= KD < 20
    if np.any(m_blue):
        ax.scatter(
            emb_2d[m_blue, 0],
            emb_2d[m_blue, 1],
            s=30,
            c="blue",
            alpha=POINT_ALPHA,
            label="0 ≤ KD < 20"
        )

    # KD >= 20
    if np.any(m_purple):
        ax.scatter(
            emb_2d[m_purple, 0],
            emb_2d[m_purple, 1],
            s=30,
            c="purple",
            alpha=POINT_ALPHA,
            label="KD ≥ 20"
        )

    ax.set_title(f"CDR: {cdr} (n={mask.sum()})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

# hide unused axes
for j in range(i + 1, nrow * ncol):
    axes[j // ncol, j % ncol].axis("off")

# -------------------------------------------------------
# legend
# -------------------------------------------------------
handles = [
    mpl.lines.Line2D([], [], linestyle="", marker="o", markersize=8,
                     markerfacecolor="red", label="KD = -1"),
    mpl.lines.Line2D([], [], linestyle="", marker="o", markersize=8,
                     markerfacecolor="blue", label="0 ≤ KD < 20"),
    mpl.lines.Line2D([], [], linestyle="", marker="o", markersize=8,
                     markerfacecolor="purple", label="KD ≥ 20"),
]

fig.legend(
    handles,
    ["KD = -1", "0 ≤ KD < 20", "KD ≥ 20"],
    loc="lower center",
    ncol=3,
    title="KD",
    bbox_to_anchor=(0.5, -0.02)
)

plt.suptitle(
    f"{pdb_for_save}",
    y=0.98
)

plt.tight_layout(rect=[0, 0.05, 1, 0.94])

out_path = os.path.join(
    save_dir,
    f"{pdb_for_save}_exp_by_CDR.png"
)
plt.savefig(out_path, dpi=200)
plt.close()

print("Saved:", out_path)
