import pandas as pd
import numpy as np
import random
from collections import defaultdict


import itertools 
import re
import copy 

import torch 
from torch.utils.data import Dataset

from data import utils as du
from data import residue_constants as rc 

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from data.motif_index import crop_antigen

import logging

def _process_mutations(processed_file_path, mut, scaffold_idx):
    processed_feats = du.read_pkl(processed_file_path)
    
    # [Safety] scaffold_idx 원본 보존을 위해 Deep Copy 수행
    updated_scaffold_idx = copy.deepcopy(scaffold_idx)
    
    # 텐서 변환
    for k, v in processed_feats.items():
        if not isinstance(v, torch.Tensor):
            processed_feats[k] = torch.tensor(v)

    if mut == 'No_Mutation':
        # 변동이 없으므로 그대로 반환
        return processed_feats, updated_scaffold_idx

    raw_mutations = mut.split('_')
    parsed_mutations = []

    # 1. Parsing
    for m in raw_mutations:
        match = re.match(r"([a-zA-Z])([a-zA-Z0-9])(\d+)(.*)", m)
        if not match:
            print(f"Warning: Cannot parse mutation string {m}")
            continue
        
        aa_char, chain_char, res_id_str, type_str = match.groups()
        target_res_id = int(res_id_str)
        target_chain_idx = du.chain_str_to_int(chain_char)
        
        parsed_mutations.append({
            'aa_char': aa_char,
            'chain_idx': target_chain_idx,
            'res_id': target_res_id,
            'type': type_str,
            'orig_str': m
        })

    # 2. Residue ID 기준 내림차순 정렬
    parsed_mutations.sort(key=lambda x: x['res_id'], reverse=True)

    # 3. Mutation 처리
    for p_mut in parsed_mutations:
        target_res_id = p_mut['res_id']
        target_chain_idx = p_mut['chain_idx']
        type_str = p_mut['type']
        aa_char = p_mut['aa_char']

        # 현재 Feature에서 절대 위치(Python Index) 찾기
        curr_chain = processed_feats['chain_index']
        curr_res = processed_feats['residue_index']
        
        mask_loc = (curr_chain == target_chain_idx) & (curr_res == target_res_id)
        idx_loc = torch.where(mask_loc)[0]
        
        if len(idx_loc) == 0:
            print(f"Warning: Target residue {p_mut['orig_str']} location not found.")
            continue
        
        idx = idx_loc[0].item()

        # --- Deletion ---
        if type_str == 'del':
            # 1. Feature Tensor 삭제
            for key in processed_feats.keys():
                feat = processed_feats[key]
                processed_feats[key] = torch.cat([feat[:idx], feat[idx+1:]], dim=0)
            
            # 2. 뒤쪽 Residue Index 조정 (-1)
            mask_update = (processed_feats['chain_index'] == target_chain_idx) & \
                        (processed_feats['residue_index'] > target_res_id)
            processed_feats['residue_index'][mask_update] -= 1

            # 3. scaffold_idx 업데이트 (updated_scaffold_idx 사용)
            for region_key in updated_scaffold_idx:
                if updated_scaffold_idx[region_key] > idx:
                    updated_scaffold_idx[region_key] -= 1

        # --- Insertion ---
        elif type_str == 'ins':
            inserted_aa_idx = rc.restype_order.get(aa_char, 20)
            insert_pos = idx + 1
            
            new_row = {}
            new_row['aatype'] = torch.tensor([inserted_aa_idx], dtype=processed_feats['aatype'].dtype)
            new_row['atom_mask'] = torch.tensor(rc.STANDARD_ATOM_MASK[inserted_aa_idx]).unsqueeze(0)
            
            new_res_id = target_res_id + 1
            new_row['residue_index'] = torch.tensor([new_res_id], dtype=processed_feats['residue_index'].dtype)
            new_row['chain_index'] = torch.tensor([target_chain_idx], dtype=processed_feats['chain_index'].dtype)
            new_row['bb_mask'] = torch.tensor([1], dtype=processed_feats['bb_mask'].dtype)
            
            new_row['atom_positions'] = torch.zeros((1, 37, 3), dtype=processed_feats['atom_positions'].dtype)
            new_row['b_factors'] = torch.zeros((1, 37), dtype=processed_feats['b_factors'].dtype)
            new_row['bb_positions'] = torch.zeros((1, 3), dtype=processed_feats['bb_positions'].dtype)
            new_row['modeled_idx'] = torch.tensor([0], dtype=processed_feats['modeled_idx'].dtype)

            # 삽입 전, 뒷부분 Residue Index들을 먼저 +1 밀어줌
            mask_update = (processed_feats['chain_index'] == target_chain_idx) & \
                        (processed_feats['residue_index'] > target_res_id)
            processed_feats['residue_index'][mask_update] += 1
            
            # 데이터 삽입
            for key in processed_feats.keys():
                curr_data = processed_feats[key]
                to_insert = new_row[key]
                processed_feats[key] = torch.cat([
                    curr_data[:insert_pos],
                    to_insert,
                    curr_data[insert_pos:]
                ], dim=0)

            # scaffold_idx 업데이트 (updated_scaffold_idx 사용)
            for region_key in updated_scaffold_idx:
                if updated_scaffold_idx[region_key] >= insert_pos:
                    updated_scaffold_idx[region_key] += 1

        # --- Substitution ---
        else:
            mutated_aa_char = type_str
            mutated_aa_idx = rc.restype_order.get(mutated_aa_char, 20)
            processed_feats['aatype'][idx] = mutated_aa_idx
            processed_feats['atom_mask'][idx] = torch.tensor(rc.STANDARD_ATOM_MASK[mutated_aa_idx])

    # 수정된 피처와 업데이트된 인덱스를 함께 반환
    return processed_feats, updated_scaffold_idx

def _process_csv_row(processed_file_path, mut, scaffold_idx):
    processed_feats = du.read_pkl(processed_file_path)
    if mut != "No_Mutation":
        processed_feats, scaffold_idx = _process_mutations(processed_file_path, mut, scaffold_idx)
    processed_feats = du.parse_chain_feats(processed_feats)

    # make chain sequence list (for the multimer relpos embedding)
    int_to_aa = {i: restype for restype, i in rc.restype_order_with_x.items()}
    aatypes = processed_feats["aatype"].tolist()            # [L]
    chain_indices = processed_feats["chain_index"].tolist()  # [L]

    chain_seqs = defaultdict(list)
    chain_order = []

    for aa_int, chain_id in zip(aatypes, chain_indices):
        aa_letter = int_to_aa.get(int(aa_int), "X")
        chain_seqs[chain_id].append(aa_letter)
        if chain_id not in chain_order:
            chain_order.append(chain_id)
    print("chain_seqs", chain_seqs)
    chain_seq_list = ["".join(chain_seqs[chain_id]) for chain_id in chain_order]

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'chain_index': torch.tensor(processed_feats['chain_index']),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).float(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).float(),
        'seq_mask': torch.tensor(processed_feats['bb_mask']).int()
    }
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles(chain_feats)
    chain_feats = data_transforms.get_chi_angles(chain_feats)
    chain_feats = data_transforms.get_backbone_frames(chain_feats)
    chain_feats['pseudo_beta'] = data_transforms.pseudo_beta_fn(
        chain_feats['aatype'],
        chain_feats['all_atom_positions'],
        None
        )
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()
    res_mask[chain_feats['aatype'] == 20] = 0
    chain_idx = torch.tensor(processed_feats['chain_index'])
    residue_index = torch.tensor(processed_feats['residue_index'])

    return {
        'res_plddt': torch.tensor(res_plddt),
        'aatype': chain_feats['aatype'],
        'chain_index': chain_feats['chain_index'],
        'res_mask': res_mask,
        'chain_idx': chain_idx,
        'residue_index': residue_index,
        'scaffold_idx': scaffold_idx,
        'chain_seq_list': chain_seq_list,
        'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        'alt_torsion_angles_sin_cos': chain_feats['alt_torsion_angles_sin_cos'],
        'torsion_angles_mask': chain_feats['torsion_angles_mask'],
        'chi_angles_sin_cos': chain_feats['chi_angles_sin_cos'],
        'chi_mask': chain_feats['chi_mask'],
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'], # (L, 14, 3)
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'], # require centering 
        'atom14_alt_gt_positions': chain_feats['atom14_alt_gt_positions'], # require centering 
        'atom14_alt_gt_exists': chain_feats['atom14_alt_gt_exists'],
        'atom14_atom_is_ambiguous': chain_feats['atom14_atom_is_ambiguous'],
        'backbone_rigid_mask': chain_feats['backbone_rigid_mask'],
        'rigidgroups_gt_frames': chain_feats['rigidgroups_gt_frames'], # require centering  (L, 8, 4, 4)
        'rigidgroups_gt_exists': chain_feats['rigidgroups_gt_exists'],
        'rigidgroups_alt_gt_frames': chain_feats['rigidgroups_alt_gt_frames'], # require centering (L, 8, 4, 4)
    }

# ==============================================================================
# 2. 데이터 전처리 및 환경 설정 클래스
# ==============================================================================
class DataManager:
    def __init__(self, dataset_cfg, is_training):
        if is_training:
            self.main_csv_path = dataset_cfg.train_csv_path
            self.meta_csv_path = dataset_cfg.train_meta_path
        else:
            self.main_csv_path = dataset_cfg.valid_csv_path 
            self.meta_csv_path = dataset_cfg.valid_meta_path 

        print(">> Loading Metadata...")
        self.chain_to_meta_row = self._load_metadata()
        
        print(">> Loading and Preprocessing Main Data...")
        self.affinity_df, self.cluster_group = self._load_main_data()
        
    def _load_metadata(self):
            """
            metadata.csv를 로드하여 {pdb_name: {row_data}} 매핑 딕셔너리 생성
            """
            meta_df = pd.read_csv(self.meta_csv_path)
            
            # 1. pdb_name을 인덱스로 설정 (Key로 사용하기 위함)
            # 2. to_dict('index')를 사용하여 {인덱스: {컬럼명: 값, ...}} 형태로 변환
            return meta_df.set_index('pdb_name').to_dict(orient='index')

    def _load_main_data(self):
        df = pd.read_csv(self.main_csv_path)
        df = df[df['Affinity_Kd [nM]'] != -2].copy()
        required_cols = ['Affinity_Kd [nM]', 'Ab_cluster_0.9', 'Ag_cluster_0.75', 'mapped_chains', 'mutation']
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        df['Affinity_Kd [nM]'] = df['Affinity_Kd [nM]'].replace(-1, float('inf'))
        df['cluster_id'] = df['Ab_cluster_0.9'].astype(str) + "_" + df['Ag_cluster_0.75'].astype(str)
        cluster_group = df.groupby('cluster_id').indices
        
        print(f"   Total Valid Samples: {len(df)}")
        print(f"   Total Unique Clusters: {len(cluster_group)}")
        return df, cluster_group

# ==============================================================================
# 3. Pair Sampler (Pairing Logic - 변수 적용됨)
# ==============================================================================

class AffinityPairSampler:
    def __init__(self, data_manager, pairs_per_cluster=6, is_training=True, seed=42):
        self.affinity_df = data_manager.affinity_df
        self.cluster_dict = data_manager.cluster_group
        self.clusters = list(self.cluster_dict.keys())
        self.kd_values = self.affinity_df['Affinity_Kd [nM]'].values
        
        self.pairs_per_cluster = pairs_per_cluster
        self.is_training = is_training
        self.seed = seed
        
        # Generator 초기화 (Training이면 None -> 랜덤, Valid면 고정 값)
        # Validation에서도 매 Epoch마다 똑같은 결과를 얻으려면 generate 함수 안에서 리셋해야 함
        self.rng = np.random.default_rng(None if is_training else seed)

    def check_kd_ratio(self, idx1, idx2):
        # ... (기존과 동일) ...
        kd1 = self.kd_values[idx1]
        kd2 = self.kd_values[idx2]

        if kd1 == float('inf') and kd2 == float('inf'): return None, False
        if kd1 == float('inf'): return 0, True
        if kd2 == float('inf'): return 1, True
        if kd1 <= 0 or kd2 <= 0: return None, False

        if kd1 >= 10 * kd2: return 0, True
        elif kd2 >= 10 * kd1: return 1, True
        else: return None, False

    def generate_epoch_pairs(self):
        # [핵심] Validation 모드일 경우, 함수 호출 시마다 시드를 리셋하여 
        # 항상 '똑같은 Pair 조합'이 나오도록 보장합니다.
        if not self.is_training:
            self.rng = np.random.default_rng(self.seed)
            print(f"   >> [Valid] Seed reset to {self.seed} for deterministic pairing.")
        else:
            # Training일 때는 계속 랜덤 상태 유지
            pass

        pairs = []
        seen_pairs = set() 
        clusters_insufficient = 0
        
        # Generator를 이용해 셔플 (이제 random.shuffle 대신 self.rng.shuffle 사용)
        # self.clusters는 원본 보존을 위해 복사 후 셔플 추천
        current_clusters = self.clusters.copy()
        self.rng.shuffle(current_clusters) 

        # ==================================================================
        # 1. Intra-Cluster Pairing
        # ==================================================================
        print(f"   >> Processing Intra-cluster pairing...")
        
        for cluster in current_clusters:
            indices = self.cluster_dict[cluster]
            n_samples = len(indices)
            
            if n_samples < 2:
                clusters_insufficient += 1
                continue

            found_for_this_cluster = 0
            
            if n_samples <= 30:
                all_combos = list(itertools.combinations(indices, 2))
                # 리스트 셔플도 rng 사용
                self.rng.shuffle(all_combos)
                
                for idx1, idx2 in all_combos:
                    pair_key = tuple(sorted((idx1, idx2)))
                    if pair_key in seen_pairs: continue

                    label, is_valid = self.check_kd_ratio(idx1, idx2)
                    if is_valid:
                        pairs.append({'idx1': idx1, 'idx2': idx2, 'label': label, 'type': 'intra'})
                        seen_pairs.add(pair_key)
                        found_for_this_cluster += 1
                        if found_for_this_cluster >= self.pairs_per_cluster: break
            
            else:
                attempts = 0
                while found_for_this_cluster < self.pairs_per_cluster and attempts < 100:
                    attempts += 1
                    # rng.choice 사용
                    idx1, idx2 = self.rng.choice(indices, 2, replace=False)
                    
                    pair_key = tuple(sorted((idx1, idx2)))
                    if pair_key in seen_pairs: continue
                    
                    label, is_valid = self.check_kd_ratio(idx1, idx2)
                    if is_valid:
                        pairs.append({'idx1': idx1, 'idx2': idx2, 'label': label, 'type': 'intra'})
                        seen_pairs.add(pair_key)
                        found_for_this_cluster += 1

            if found_for_this_cluster < self.pairs_per_cluster:
                clusters_insufficient += 1

        # ==================================================================
        # 2. Inter-Cluster Pairing
        # ==================================================================
        print("   >> Processing Inter-cluster pairing...")
        
        for cluster_a in current_clusters:
            indices_a = self.cluster_dict[cluster_a]
            idx1 = self.rng.choice(indices_a) # rng 사용
            
            for _ in range(20):
                cluster_b = self.rng.choice(current_clusters) # rng 사용
                if cluster_a == cluster_b: continue
                
                indices_b = self.cluster_dict[cluster_b]
                idx2 = self.rng.choice(indices_b) # rng 사용
                
                pair_key = tuple(sorted((idx1, idx2)))
                if pair_key in seen_pairs: continue
                
                label, is_valid = self.check_kd_ratio(idx1, idx2)
                if is_valid:
                    pairs.append({'idx1': idx1, 'idx2': idx2, 'label': label, 'type': 'inter'})
                    seen_pairs.add(pair_key)
                    break
        
        return pd.DataFrame(pairs)

# ==============================================================================
# 4. Dataset 정의 (기존 동일)
# ==============================================================================
class AffinityDataset(Dataset):
    def __init__(self, data_manager, pair_df):
        self.main_df = data_manager.affinity_df
        self.pair_df = pair_df
        self.meta_row_mapping = data_manager.chain_to_meta_row

    def __len__(self):
        return len(self.pair_df)

    def _get_single_sample(self, idx):
        row = self.main_df.iloc[idx]
        chains = str(row['mapped_chains']).split(';')
        mutations = str(row['mutation']).split(';')
        
        if len(chains) != len(mutations):
            part_idx = 0
        else:
            part_idx = random.randint(0, len(chains) - 1)
            
        selected_chain = chains[part_idx].strip()
        selected_mutation = mutations[part_idx].strip()
        selected_meta_row = self.meta_row_mapping[selected_chain]
        
        return selected_mutation, selected_meta_row


    def process_csv_row(self, csv_row, mut):
        path = csv_row['processed_path']
        scaffold_idx = {}
        cdr_types = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']
        for cdr in cdr_types:
            scaffold_idx[f'{cdr}_start'] = int(csv_row[f'{cdr}_start'])
            scaffold_idx[f'{cdr}_end'] = int(csv_row[f'{cdr}_end'])

        processed_row = _process_csv_row(path, mut, scaffold_idx)
        processed_row['mode'] = csv_row['mode']
        processed_row['raw_path'] = csv_row['raw_path'] 

        return processed_row

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        self.csv = data_csv
        if self.is_training:
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            self._log.info(
                f'Validation: {len(self.csv)} examples')
            
        self.csv['index'] = list(range(len(self.csv)))

    def _sample_scaffold_mask(self, batch):
        aatype = batch['aatype']
        mode = batch['mode']
        num_res = aatype.shape[0]
        scaffold_idx = batch['scaffold_idx']
        scaffold_mask = torch.zeros(num_res)

        if mode == 'ab': # antibody-antigen
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(6):
                scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1.0

        if mode == 'nanobody': # Nanobody-antigen
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(3):
                scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1.0

        return scaffold_mask * batch['res_mask']

    def setup_inpainting(self, feats):
        loop_mask = self._sample_scaffold_mask(feats)
        feats['loop_mask'] = loop_mask

    def __getitem__(self, i):
        pair_row = self.pair_df.iloc[i]
        label = pair_row['label']
        feats_paired = {}

        for sample_num in range(2):
            mut, meta_row = self._get_single_sample(pair_row[f'idx{sample_num+1}'])
            feats = self.process_csv_row(meta_row, mut)
        
            rigids_1 = rigid_utils.Rigid.from_tensor_4x4(feats['rigidgroups_gt_frames'])[:, 0]
            feats['rotmats_1'] = torch.tensor(rigids_1.get_rots().get_rot_mats(), device=rigids_1.device)
            feats['trans_1'] = torch.tensor(rigids_1.get_trans(), device=rigids_1.device)

            self.setup_inpainting(feats)
            # # modify loop mask (if it is terminal mask, exclude end residue to provide an anchor)
            # feats['loop_mask'] = provide_anchor(
            #     feats['loop_mask'], 
            #     feats['res_mask'], 
            #     feats['chain_index'],
            #     feats['mode']
            #     ).to(torch.long)
            
            # make diffuse_mask             
            chain_len_list = [len(seq) for seq in feats['chain_seq_list']]
            if len(chain_len_list) == 1:
                diffuse_mask = feats['loop_mask']
            else:
                asym_id = []
                for chain_idx, chain_len in enumerate(chain_len_list):
                    for _ in range(chain_len):
                        asym_id.append(chain_idx)
                asym_id = torch.tensor(asym_id, device=feats['loop_mask'].device)
                print("asym_id", asym_id.shape)
                print("loop_mask", feats['loop_mask'].shape)
                masked_chain = asym_id[feats['loop_mask'] == 1].unique()
                diffuse_mask = torch.isin(asym_id, masked_chain).to(torch.long)
            feats['diffuse_mask'] = diffuse_mask

            feats['crop_idx'] = crop_antigen(
                feats['trans_1'],
                cdr_mask=feats['loop_mask'],
                nan_mask=feats['res_mask'],
                max_len=self.dataset_cfg.ab_max_num_res,
                seq_list=feats['chain_seq_list'],
                crop_ab=self.dataset_cfg.crop_ab,
                mode=feats['mode'],
                )
            feats['loop_mask'] = feats['loop_mask'].int()
            feats_paired[f"batch_{sample_num}"] = feats
        feats_paired['label'] = label
        
        return feats_paired

class PdbDataset(AffinityDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self.dataset_cfg = dataset_cfg
        self.task = task
        self.current_epoch = 0

        # ------------------------------------------------------------------
        # 1. DataManager & Sampler 초기화
        # ------------------------------------------------------------------
        # 데이터를 로드합니다.
        self.data_manager = DataManager(dataset_cfg, is_training)
        
        # Sampler를 초기화합니다.

        # ------------------------------------------------------------------
        # 2. 초기 Pair 생성
        # ------------------------------------------------------------------
        # 학습 초기 Pair를 생성합니다.
        if self._is_training:
            print(f">> [Init] Generating initial pairs for training...")
            self.sampler = AffinityPairSampler(
                self.data_manager, 
                pairs_per_cluster=dataset_cfg.pairs_per_cluster,
                is_training=True
            )
            initial_pairs = self.sampler.generate_epoch_pairs()
        else:
            print(f">> [Init] Generating pairs for validation...")
            self.sampler = AffinityPairSampler(
                self.data_manager, 
                pairs_per_cluster=dataset_cfg.pairs_per_cluster,
                is_training=False
            )
            initial_pairs = self.sampler.generate_epoch_pairs()
        # ------------------------------------------------------------------
        # 3. 부모 클래스 (AffinityDataset) 초기화
        # ------------------------------------------------------------------
        super().__init__(self.data_manager, initial_pairs)
        
        # AffinityDataset의 __getitem__에서 crop_antigen 호출 시 
        # self.dataset_cfg에 접근하므로 여기서 확실히 할당해둡니다.
        self.dataset_cfg = dataset_cfg 

        self._log.info(f'{("Training" if is_training else "Validation")} Dataset initialized.')
        self._log.info(f'Total Pairs: {len(self.pair_df)}')


    def set_current_epoch(self, epoch):
        """
        Trainer에서 매 Epoch 시작 시 호출해주어야 합니다.
        새로운 Epoch마다 Pair를 다시 샘플링하여 데이터 다양성을 확보합니다.
        """
        self.current_epoch = epoch
        
        # 학습 모드일 때만 매 Epoch마다 Pair를 섞어줍니다.
        if self._is_training:
            self._log.info(f">> [Epoch {epoch}] Regenerating pairs for diversity...")
            
            # 1. 새로운 Pair 생성
            new_pairs = self.sampler.generate_epoch_pairs()
            
            # 2. 데이터셋 내부의 pair_df 교체
            self.pair_df = new_pairs
            self._log.info(f">> [Epoch {epoch}] Pair regeneration complete. Total pairs: {len(self.pair_df)}")