import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

# ============================================================
# 1️⃣ CSV load + 전처리
# ============================================================
csv_file = '/home/psh/protein-frame-flow/inference_outputs/CDRFlow_v2.3.2.1_stage_2_wt_confidence_lr_1e-3/2025-10-26_10-47-26/epoch=42-step=61619/6wgl_insert/plddt/plddt.csv'
pdb_id = '6wgl_A_B_C'
base_dir = os.path.dirname(os.path.dirname(csv_file))

df = pd.read_csv(csv_file)
df['h_avg'] = df[['h1_plddt', 'h2_plddt', 'h3_plddt']].mean(axis=1)
best_rows = df.loc[df.groupby('mt_id')['h_avg'].idxmax()].reset_index(drop=True)
best_rows['pure_mt_id'] = best_rows['mt_id'].str.replace(f'{pdb_id}_', '', regex=False)

def extract_numbers(x):
    if x == pdb_id:
        return []
    return [int(p[:-1]) for p in x.split('_')]
    

best_rows['num_list'] = best_rows['pure_mt_id'].apply(extract_numbers)
best_rows['h1_count'] = best_rows['num_list'].apply(lambda x: sum(25 <= n <= 32 for n in x))
best_rows['h2_count'] = best_rows['num_list'].apply(lambda x: sum(51 <= n <= 57 for n in x))

# ============================================================
# 2️⃣ RMSD 벡터화 함수
# ============================================================
def load_rmsd_vector(rmsd_path):
    with open(rmsd_path, 'r') as f:
        rmsd_info = json.load(f)
    vec_h1 = np.array(rmsd_info['h1_rmsd'])
    vec_h2 = np.array(rmsd_info['h2_rmsd'])
    vec = np.concatenate([vec_h1, vec_h2], axis=0)
    return vec

# ============================================================
# 3️⃣ 그룹별 처리 함수 (t-SNE 버전)
# ============================================================
def process_group(group_tuple):
    (h1c, h2c), group = group_tuple
    rmsd_vectors = []
    valid_indices = []

    for idx, row in group.iterrows():    
        rmsd_path = os.path.join(base_dir, row['mt_id'], row['sample_id'], "rmsd_info.json")

        if not os.path.exists(rmsd_path):
            continue
        vec = load_rmsd_vector(rmsd_path)   # (n_residues, 3)
        rmsd_vectors.append(vec.flatten())  # 1D로 변환
        valid_indices.append(idx)

    # 이제 rmsd_vectors는 [(n_res*3,), (n_res*3,), ...] 형태
    X = np.vstack(rmsd_vectors)  # (n_samples, n_res*3)    

    # 차원 축소 (t-SNE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1))
    X_emb = tsne.fit_transform(X)
    
    # 클러스터링
    if h1c + h2c == 4:
        n_clusters = 50
    elif h1c + h2c == 3:
        n_clusters = 25
    elif h1c + h2c == 2:
        n_clusters = 10
    elif h1c + h2c == 1:
        n_clusters = len(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_emb)

    # n_clusters 개수에 맞는 colormap 생성
    colors = cm.get_cmap('Spectral', n_clusters)  # tab10, Set1 등 사용 가능


    plt.figure(figsize=(6,5))
    for cl in range(n_clusters):
        plt.scatter(X_emb[labels==cl, 0], X_emb[labels==cl, 1], 
                    color=colors(cl), label=f'Cluster {cl}', alpha=0.7, edgecolors='k', linewidths=0.5)

    plt.title(f"h1_count={h1c}, h2_count={h2c}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # if h1c + h2c != 4:
    #     plt.legend(loc='best', fontsize=9)  # 'best'는 point를 가리지 않는 위치 자동 선택
    plt.tight_layout()
    plt.show()
    
    # PNG 저장 경로
    out_dir = os.path.join(base_dir, "plddt", "tsne_plots")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"h1_{h1c}_h2_{h2c}.png")
    plt.savefig(png_path)
    plt.close()

    # 결과 DataFrame에 cluster 추가
    result_group = group.loc[valid_indices].copy()
    result_group['cluster_id'] = labels
    return result_group

# ============================================================
# 4️⃣ 멀티프로세싱
# ============================================================
cluster_results = []
print(best_rows)

with ProcessPoolExecutor(max_workers=1) as executor:
    futures = []
    for g in best_rows.groupby(['h1_count', 'h2_count']):
        (h1c, h2c), _ = g[0], g[1]

        if h1c == 0 and h2c == 0:
            continue  # h1_count, h2_count 모두 0이면 스킵
        futures.append(executor.submit(process_group, g))

    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            cluster_results.append(res)

# 최종 CSV 저장
if cluster_results:
    final_df = pd.concat(cluster_results, ignore_index=True)
    out_csv = os.path.join(base_dir, 'plddt', 'clustered_results.csv')
    final_df.to_csv(out_csv, index=False)
    print("Clustered CSV saved to:", out_csv)
else:
    print("No clustering results")
