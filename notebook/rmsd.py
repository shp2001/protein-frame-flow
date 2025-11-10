import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# =======================
# 1️⃣ 데이터 로드 및 준비
# =======================
input_csv = '/home/psh/cdr_bfactor_h3_rmsd_samples.csv'
output_metric_csv = '/home/psh/cdr_plddt_metric_with_pdb_corr.csv'
output_weight_csv = '/home/psh/plddt_feature_weights.csv'

df = pd.read_csv(input_csv)

features = [f'{cdr}_plddt' for cdr in ['h1','h2','h3','l1','l2','l3']]
target = 'h3_rmsd'

df = df.dropna(subset=features + [target])
print(f"✅ 유효 샘플 수: {len(df)}")

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values)
y = df[target].values

# =======================
# 2️⃣ 하위 RMSD 강조 (가중 목적)
# =======================
threshold = np.percentile(y, 25)
low_mask = y <= threshold

# =======================
# 3️⃣ 목적 함수 정의
# =======================
def metric_score(weights):
    metric = X.dot(weights)
    corr_full, _ = spearmanr(metric, y)
    corr_low, _ = spearmanr(metric[low_mask], y[low_mask])
    # 낮은 RMSD(좋은 샘플)에 더 높은 가중치
    score = 0.7 * corr_low + 0.3 * corr_full
    return -score  # maximize → minimize(-score)

# =======================
# 4️⃣ Differential Evolution 최적화
# =======================
bounds = [(-2, 2)] * len(features)
result = differential_evolution(
    metric_score,
    bounds,
    maxiter=300,
    popsize=30,
    tol=1e-6,
    polish=True,
    seed=42,
    updating="deferred",  # 병렬 최적화
    workers=-1             # 모든 CPU 코어 사용
)

opt_weights = result.x / np.linalg.norm(result.x)
df['metric'] = X.dot(opt_weights)

# =======================
# 5️⃣ 전체 / 하위 25% 상관관계 평가
# =======================
corr_full, _ = spearmanr(df['metric'], y)
corr_low, _ = spearmanr(df.loc[low_mask, 'metric'], df.loc[low_mask, target])

print("\n✅ 최적화 완료!")
for f, w in zip(features, opt_weights):
    print(f"{f:<20} {w:+.3f}")
print(f"\n📈 전체 Spearman: {corr_full:.3f}")
print(f"🎯 하위 25% Spearman: {corr_low:.3f}")

# =======================
# 6️⃣ 각 PDB별 Spearman 계산 (병렬)
# =======================
def compute_pdb_corr(pdb_id, sub_df):
    """개별 pdb의 Spearman 계산 함수"""
    if len(sub_df) < 3:
        return pdb_id, np.nan
    corr, _ = spearmanr(sub_df['metric'], sub_df[target])
    return pdb_id, corr

pdb_groups = list(df.groupby('pdb_id'))

print(f"⚙️ {len(pdb_groups)}개 PDB에 대해 Spearman 계산 중... (multi-CPU)")

spearman_results = []
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(compute_pdb_corr, pdb, sub): pdb for pdb, sub in pdb_groups}
    for future in as_completed(futures):
        pdb_id, corr = future.result()
        spearman_results.append((pdb_id, corr))

spearman_df = pd.DataFrame(spearman_results, columns=['pdb_id', 'pdb_spearman'])
df_with_corr = df.merge(spearman_df, on='pdb_id', how='left')

# =======================
# 7️⃣ 결과 저장
# =======================
df_with_corr.to_csv(output_metric_csv, index=False)
print(f"💾 Spearman 결과 저장: {output_metric_csv}")

weight_df = pd.DataFrame({
    'feature': features,
    'weight': opt_weights
})
weight_df.to_csv(output_weight_csv, index=False)
print(f"💾 가중치 저장: {output_weight_csv}")

print("\n✅ 모든 작업 완료!")
