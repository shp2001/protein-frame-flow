import pandas as pd
import numpy as np
import itertools
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


# ============================================================
# 1. 최소 DataManager (pairing 통계용)
# ============================================================

class DataManager:

    def __init__(self, csv_path):
        self.main_csv_path = csv_path

        self.affinity_df, self.cluster_group = self._load_main_data()

    def _load_main_data(self):

        df = pd.read_csv(self.main_csv_path)

        df = df[df['Affinity_Kd [nM]'] != -2].copy()

        if 'cluster' in df.columns:
            required_cols = [
                'Affinity_Kd [nM]',
                'cluster',
                'mapped_chains',
                'mutation'
            ]
        else:
            required_cols = [
                'Affinity_Kd [nM]',
                'Ab_cluster_0.9',
                'Ag_cluster_0.75',
                'mapped_chains',
                'mutation'
            ]

        df = df.dropna(subset=required_cols).reset_index(drop=True)

        df['Affinity_Kd [nM]'] = df['Affinity_Kd [nM]'].replace(-1, float('inf'))

        if 'cluster' in df.columns:
            df['cluster_id'] = df['cluster'].astype(str)
        else:
            df['cluster_id'] = (
                df['Ab_cluster_0.9'].astype(str)
                + "_"
                + df['Ag_cluster_0.75'].astype(str)
            )

        cluster_group = df.groupby('cluster_id').indices

        return df, cluster_group


# ============================================================
# 2. AffinityPairSampler
# ============================================================

class AffinityPairSampler:

    def __init__(self, data_manager, pairs_per_cluster=6, is_training=True, seed=42):

        self.affinity_df = data_manager.affinity_df
        self.cluster_dict = data_manager.cluster_group
        self.clusters = list(self.cluster_dict.keys())
        self.kd_values = self.affinity_df['Affinity_Kd [nM]'].values

        self.pairs_per_cluster = pairs_per_cluster
        self.is_training = is_training
        self.seed = seed

        self.rng = np.random.default_rng(None if is_training else seed)

    def check_kd_ratio(self, idx1, idx2, threshold):

        kd1 = self.kd_values[idx1]
        kd2 = self.kd_values[idx2]

        if kd1 == float('inf') and kd2 == float('inf'):
            return None, False

        if kd1 == float('inf'):
            return 0.0, True

        if kd2 == float('inf'):
            return 1.0, True

        if kd1 <= 0 or kd2 <= 0:
            return None, False

        if kd1 >= threshold * kd2:
            return 0.0, True
        elif kd2 >= threshold * kd1:
            return 1.0, True
        else:
            return None, False

    def generate_epoch_pairs(self, epoch):

        if not self.is_training:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng(self.seed + epoch)

        pairs = []
        seen_pairs = set()

        current_clusters = self.clusters.copy()
        self.rng.shuffle(current_clusters)

        # -------------------------------
        # 1. intra cluster
        # -------------------------------

        if self.pairs_per_cluster > 0:

            for cluster in current_clusters:

                indices = self.cluster_dict[cluster]
                n_samples = len(indices)

                if n_samples < 2:
                    continue

                found = 0

                if n_samples <= 30:

                    all_combos = list(itertools.combinations(indices, 2))
                    self.rng.shuffle(all_combos)

                    for idx1, idx2 in all_combos:

                        pair_key = tuple(sorted((idx1, idx2)))
                        if pair_key in seen_pairs:
                            continue

                        label, is_valid = self.check_kd_ratio(
                            idx1, idx2, threshold=10
                        )

                        if is_valid:
                            pairs.append({
                                "idx1": idx1,
                                "idx2": idx2,
                                "label": label,
                                "kd1": self.kd_values[idx1],
                                "kd2": self.kd_values[idx2],
                                "type": "intra"
                            })
                            seen_pairs.add(pair_key)
                            found += 1

                            if found >= self.pairs_per_cluster:
                                break

                else:

                    attempts = 0
                    while found < self.pairs_per_cluster and attempts < 100:

                        attempts += 1

                        idx1, idx2 = self.rng.choice(indices, 2, replace=False)

                        pair_key = tuple(sorted((idx1, idx2)))
                        if pair_key in seen_pairs:
                            continue

                        label, is_valid = self.check_kd_ratio(
                            idx1, idx2, threshold=10
                        )

                        if is_valid:
                            pairs.append({
                                "idx1": idx1,
                                "idx2": idx2,
                                "label": label,
                                "kd1": self.kd_values[idx1],
                                "kd2": self.kd_values[idx2],
                                "type": "intra"
                            })
                            seen_pairs.add(pair_key)
                            found += 1

        # -------------------------------
        # 2. inter cluster
        # -------------------------------

        for cluster_a in current_clusters:

            indices_a = self.cluster_dict[cluster_a]
            idx1 = self.rng.choice(indices_a)

            for _ in range(20):

                cluster_b = self.rng.choice(current_clusters)
                if cluster_a == cluster_b:
                    continue

                indices_b = self.cluster_dict[cluster_b]
                idx2 = self.rng.choice(indices_b)

                pair_key = tuple(sorted((idx1, idx2)))
                if pair_key in seen_pairs:
                    continue

                label, is_valid = self.check_kd_ratio(
                    idx1, idx2, threshold=10
                )

                if is_valid:
                    pairs.append({
                        "idx1": idx1,
                        "idx2": idx2,
                        "label": label,
                        "kd1": self.kd_values[idx1],
                        "kd2": self.kd_values[idx2],
                        "type": "inter"
                    })
                    seen_pairs.add(pair_key)
                    break

        return pd.DataFrame(pairs)


# ============================================================
# 3. Dataset
# ============================================================

class AffinityDataset(Dataset):

    def __init__(self, data_manager, pair_df):
        self.main_df = data_manager.affinity_df
        self.pair_df = pair_df

    def __len__(self):
        return len(self.pair_df)

    def __getitem__(self, idx):
        return self.pair_df.iloc[idx]


# ============================================================
# 4. KD ratio 수집
# ============================================================

def collect_ratios(data_manager, pairs_per_cluster, num_epochs=100):

    sampler = AffinityPairSampler(
        data_manager,
        pairs_per_cluster=pairs_per_cluster,
        is_training=True,
        seed=42,
    )

    ratios = []

    for epoch in range(num_epochs):

        pair_df = sampler.generate_epoch_pairs(epoch)

        for _, row in pair_df.iterrows():

            kd1 = row["kd1"]
            kd2 = row["kd2"]

            if not np.isfinite(kd1) or not np.isfinite(kd2):
                continue

            if kd1 <= 0 or kd2 <= 0:
                continue

            ratios.append(max(kd1, kd2) / min(kd1, kd2))

    return np.asarray(ratios)


# ============================================================
# 5. cluster 등장 빈도
# ============================================================

def collect_cluster_frequencies(data_manager, pairs_per_cluster, num_epochs=40):

    sampler = AffinityPairSampler(
        data_manager,
        pairs_per_cluster=pairs_per_cluster,
        is_training=True,
        seed=42,
    )

    cluster_ids = data_manager.affinity_df["cluster_id"].values

    counter = Counter()

    for epoch in range(num_epochs):

        pair_df = sampler.generate_epoch_pairs(epoch)

        for _, row in pair_df.iterrows():
            i1 = row["idx1"]
            i2 = row["idx2"]

            c1 = cluster_ids[i1]
            c2 = cluster_ids[i2]

            counter[c1] += 1
            counter[c2] += 1

    return counter


# ============================================================
# 6. cluster별 unique pair 수집 (★추가)
# ============================================================

def collect_unique_pairs_per_cluster(
    data_manager,
    pairs_per_cluster,
    num_epochs=40
):

    sampler = AffinityPairSampler(
        data_manager,
        pairs_per_cluster=pairs_per_cluster,
        is_training=True,
        seed=42,
    )

    cluster_ids = data_manager.affinity_df["cluster_id"].values

    # cluster_id -> set( unordered pair )
    cluster_pair_sets = defaultdict(set)

    for epoch in range(num_epochs):

        pair_df = sampler.generate_epoch_pairs(epoch)

        for _, row in pair_df.iterrows():

            i1 = int(row["idx1"])
            i2 = int(row["idx2"])

            pair_key = tuple(sorted((i1, i2)))

            c1 = cluster_ids[i1]
            c2 = cluster_ids[i2]

            # intra
            if c1 == c2:
                cluster_pair_sets[c1].add(pair_key)
            else:
                # inter pair는 양쪽 cluster에 모두 귀속
                cluster_pair_sets[c1].add(pair_key)
                cluster_pair_sets[c2].add(pair_key)

    return cluster_pair_sets


# ============================================================
# 7. main
# ============================================================

if __name__ == "__main__":

    MAIN_CSV_PATH = "/home/psh/data/affinity/ppb_affinity/affinity_train_ppb.csv"
    PAIRS_PER_CLUSTER = 0
    NUM_EPOCHS = 40

    print("loading csv ...")
    dm = DataManager(MAIN_CSV_PATH)

    print("collecting kd ratios ...")
    ratios = collect_ratios(
        dm,
        PAIRS_PER_CLUSTER,
        NUM_EPOCHS
    )

    print("num pairs :", len(ratios))
    print("min :", ratios.min())
    print("median :", np.median(ratios))
    print("max :", ratios.max())

    plt.figure(figsize=(7, 5))
    plt.hist(ratios, bins=50)
    plt.xlabel("max(kd1, kd2) / min(kd1, kd2)")
    plt.ylabel("count")
    plt.title("KD ratio distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.hist(np.log10(ratios), bins=50)
    plt.xlabel("log10( max(kd1, kd2) / min(kd1, kd2) )")
    plt.ylabel("count")
    plt.title("log10 KD ratio distribution")
    plt.tight_layout()
    plt.savefig(
        "/home/psh/protein-frame-flow/notebook/check_affinity_pair.png"
    )

    # ------------------------------------------------
    # cluster 등장 빈도
    # ------------------------------------------------

    counter = collect_cluster_frequencies(
        dm,
        PAIRS_PER_CLUSTER,
        NUM_EPOCHS
    )

    freqs = np.array(list(counter.values()))

    print("number of clusters appeared:", len(freqs))
    print("min freq:", freqs.min())
    print("median freq:", np.median(freqs))
    print("max freq:", freqs.max())

    plt.figure(figsize=(7, 5))
    plt.hist(freqs, bins=50)
    plt.xlabel("appearance count per cluster")
    plt.ylabel("number of clusters")
    plt.title("Cluster appearance frequency (over epochs)")
    plt.tight_layout()
    plt.savefig(
        "/home/psh/protein-frame-flow/notebook/check_affinity_cluster.png"
    )

    # ------------------------------------------------
    # ★ cluster별 unique pair 분포
    # ------------------------------------------------

    cluster_pair_sets = collect_unique_pairs_per_cluster(
        dm,
        PAIRS_PER_CLUSTER,
        NUM_EPOCHS
    )

    unique_pair_counts = np.array(
        [len(v) for v in cluster_pair_sets.values()]
    )

    print("\n[unique pairs per cluster]")
    print("number of clusters:", len(unique_pair_counts))
    print("min:", unique_pair_counts.min())
    print("median:", np.median(unique_pair_counts))
    print("max:", unique_pair_counts.max())

    plt.figure(figsize=(7, 5))
    plt.hist(unique_pair_counts, bins=50)
    plt.xlabel("number of unique pairs per cluster")
    plt.ylabel("number of clusters")
    plt.title("Unique pair count per cluster (over epochs)")
    plt.tight_layout()
    plt.savefig(
        "/home/psh/protein-frame-flow/notebook/check_affinity_unique_pairs_per_cluster.png"
    )
