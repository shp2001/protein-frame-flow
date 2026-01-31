import pandas as pd
import numpy as np
import random
from collections import defaultdict

import itertools 
import re
import copy 
import json 
import os 

import torch 
from torch.utils.data import Dataset

from data import utils as du
from data import residue_constants as rc 

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from data.motif_index import load_general_mask_all, crop_antigen, crop_general_protein, provide_anchor, remask_antigen_mask

import logging

def _process_mutations(processed_file_path, mut):
    processed_feats = du.read_pkl(processed_file_path)
    for k, v in processed_feats.items():
        if not isinstance(v, torch.Tensor):
            processed_feats[k] = torch.tensor(v)

    if mut == 'No_Mutation':
        return processed_feats

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

        # --- Insertion ---

        # --- Substitution ---
        mutated_aa_char = type_str
        mutated_aa_idx = rc.restype_order.get(mutated_aa_char, 20)
        processed_feats['aatype'][idx] = mutated_aa_idx
        processed_feats['atom_mask'][idx] = torch.tensor(rc.STANDARD_ATOM_MASK[mutated_aa_idx])

    # 수정된 피처와 업데이트된 인덱스를 함께 반환
    return processed_feats

def _process_csv_row(processed_file_path, mut):
    processed_feats = du.read_pkl(processed_file_path)
    if mut != "No_Mutation":
        processed_feats = _process_mutations(processed_file_path, mut)
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

        self.dataset_cfg = dataset_cfg

        self.chain_to_meta_row = self._load_metadata()
        self.affinity_df, self.cluster_group = self._load_main_data()

    def _load_metadata(self):
        """
        metadata.csv를 로드하여 {(pdb_name, data_source): {row_data}} 매핑 딕셔너리 생성
        """
        meta_df = pd.read_csv(self.meta_csv_path)
        
        # processed_path와 pdb_name 두 개의 열을 리스트로 묶어 인덱스로 설정
        if 'data_source' in meta_df.columns:
            return meta_df.set_index(['pdb_name', 'data_source']).to_dict(orient='index')
        else:
            return meta_df.set_index(['pdb_name']).to_dict(orient='index')

    def _filter_by_metadata(self, df):
        """
        Main DataFrame을 필터링합니다.
        각 행에 대해 메타데이터(길이, 체인 수 등)와 마스크 파일(유효 잔기 수)을 검증합니다.
        """
        original_len = len(df)
        
        # 모든 검증 로직(메타데이터 + 마스크파일)을 _check_row 함수 하나로 통합하여 적용
        valid_meta_mask = df.apply(self._check_row, axis=1)
        
        filtered_df = df[valid_meta_mask].reset_index(drop=True)
        
        dropped_count = original_len - len(filtered_df)
        if dropped_count > 0:
            print(f">> [Filter] Dropped {dropped_count} rows. (Total kept: {len(filtered_df)})")
            
        return filtered_df

    def _check_row(self, row):
        """
        단일 행에 대해 다음을 순차적으로 검증합니다:
        1. Metadata Check: seq_len, num_chains (In-Memory, Fast)
        2. Mask File Check: total_residues (Disk I/O, Slow)
        """
        # ---------------------------------------------------------
        # 1. Metadata Check (In-Memory)
        # ---------------------------------------------------------
        chains = str(row['mapped_chains']).split(';')
        has_source = 'Source Data Set' in row.index
        source_dataset = row['Source Data Set'] if has_source else None

        # 체인별 메타데이터 검증
        for chain_id in chains:
            chain_key = chain_id.strip()
            if has_source:
                chain_key = (chain_key, source_dataset)
            
            # 로드해둔 메타데이터 딕셔너리에서 조회
            meta_info = self.chain_to_meta_row.get(chain_key)
            
            if meta_info is None:
                # 메타데이터가 없으면 유효하지 않은 데이터로 간주
                return False 
            
            # 설정된 최대 길이/체인 수 초과 시 제외
            if meta_info['seq_len'] > self.dataset_cfg.filter.max_num_res:
                return False
            if meta_info['num_chains'] > self.dataset_cfg.filter.max_num_chain:
                return False

        if has_source:
            path = meta_info['processed_path']
            mask_path = path.replace("meta", "mask_index").replace('.pkl', '.json')

            with open(mask_path, 'r') as f:
                mask_data = json.load(f)
            
            total_residues = 0
            for chain_blocks in mask_data.values():
                for block in chain_blocks:
                    total_residues += len(block)

            if total_residues > self.dataset_cfg.filter.max_mask_residues:
                return False

        return True

    def _load_main_data(self):
        df = pd.read_csv(self.main_csv_path)

        # Affinity 필터
        df = df[df['Affinity_Kd [nM]'] != -2].copy()

        # --- cluster 컬럼 존재 여부에 따른 분기 ---
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
        df = self._filter_by_metadata(df)

        # -1 → inf 처리
        df['Affinity_Kd [nM]'] = df['Affinity_Kd [nM]'].replace(-1, float('inf'))

        # --- cluster_id 생성 ---
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
        if kd1 == float('inf'): return 0.0, True
        if kd2 == float('inf'): return 1.0, True
        if kd1 <= 0 or kd2 <= 0: return None, False

        if kd1 >= 10 * kd2: return 0.0, True
        elif kd2 >= 10 * kd1: return 1.0, True
        else: return None, False

    def generate_epoch_pairs(self):
        # [핵심] Validation 모드일 경우, 함수 호출 시마다 시드를 리셋하여 
        # 항상 '똑같은 Pair 조합'이 나오도록 보장합니다.
        if not self.is_training:
            self.rng = np.random.default_rng(self.seed)
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
                        pairs.append({'idx1': idx1, 'idx2': idx2, 'label': label, 'kd1': self.kd_values[idx1], 'kd2': self.kd_values[idx2], 'type': 'intra'})
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
                        pairs.append({'idx1': idx1, 'idx2': idx2, 'label': label, 'kd1': self.kd_values[idx1], 'kd2': self.kd_values[idx2], 'type': 'intra'})
                        seen_pairs.add(pair_key)
                        found_for_this_cluster += 1

            if found_for_this_cluster < self.pairs_per_cluster:
                clusters_insufficient += 1

        # ==================================================================
        # 2. Inter-Cluster Pairing
        # ==================================================================
        
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
                    pairs.append({'idx1': idx1, 'idx2': idx2, 'label': label, 'kd1': self.kd_values[idx1], 'kd2': self.kd_values[idx2], 'type': 'inter'})
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
        part_idx = random.randint(0, len(chains) - 1)
            
        meta_key = chains[part_idx].strip()
        selected_mutation = mutations[part_idx].strip()
        has_source = 'Source Data Set' in row.index
        if has_source:
            source_dataset = row['Source Data Set']
            meta_key = (meta_key, source_dataset)
        selected_meta_row = self.meta_row_mapping[meta_key]
        
        return selected_mutation, selected_meta_row

    def process_csv_row(self, csv_row, mut):
        path = csv_row['processed_path']
        complex_id = csv_row['pdb_name']
        processed_row = _process_csv_row(path, mut)
        
        mask_path = path.replace("meta", "mask_index").replace('.pkl', '.json')
        complex_id = os.path.basename(mask_path).replace('.json', '')
        pdb_id, ligand_chains_str, receptor_chains_str = complex_id.split('_')
        self.ligand_chains = list(ligand_chains_str)
        self.receptor_chains = list(receptor_chains_str)
        # TCR ligand-receptor mapping is changed 
        if "TCR" in csv_row['mode']:
            self.ligand_chains = list(receptor_chains_str)
            self.receptor_chains = list(ligand_chains_str)

        scaffold_mask = load_general_mask_all(
            mask_info_file=mask_path,
            residue_index=processed_row['residue_index'],
            chain_index=processed_row['cahin_index'],
            threshold_length=25,
        )
        processed_row['loop_mask'] = torch.tensor(scaffold_mask)
        processed_row['mode'] = csv_row['mode']
        processed_row['raw_path'] = csv_row['raw_path'] 
        processed_row['selected_chains'] = self.ligand_chains

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
            
            # make diffuse_mask             
            ligand_chain_indices = [du.chain_str_to_int(c) for c in self.ligand_chains]
            ligand_chain_tensor = torch.tensor(ligand_chain_indices, device=feats['loop_mask'].device)
            chain_index = feats['chain_index']
            
            ligand_mask = torch.isin(chain_index, ligand_chain_tensor).to(torch.long)
            diffuse_mask = torch.max(ligand_mask, feats['loop_mask'])
            
            # 4. 결과 저장
            feats['ligand_mask'] = ligand_mask

            diffuse_mask = provide_anchor(
                diffuse_mask, 
                feats['res_mask'], 
                feats['chain_index'],
                feats['mode']
                ).to(torch.long)
            
            feats['diffuse_mask'] = diffuse_mask
            if torch.sum(diffuse_mask) == 0:
                raise ValueError(
                    f"diffuse_mask is all zero for sample {feats['pdb_name']}. "
                    "Check if ligand_chains or loop_mask (CDR/Interface) are correctly defined."
                )

            mask_path = meta_row['processed_path'].replace("meta", "mask_index").replace('.pkl', '.json')

            feats['crop_idx'] = crop_general_protein(
                trans_1=feats['trans_1'],
                loop_mask=feats['loop_mask'],
                nan_mask=feats['res_mask'],
                max_len=self.dataset_cfg.general_max_num_res,
            )
            feats_paired[f"batch_{sample_num}"] = feats
        feats_paired['label'] = label
        feats_paired['kd1'] = pair_row['kd1']
        feats_paired['kd2'] = pair_row['kd2']
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
        self._log_pair_stats()


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

    # [새로 추가할 헬퍼 함수]
    def _log_pair_stats(self):
        total = len(self.pair_df)
        if total > 0 and 'type' in self.pair_df.columns:
            counts = self.pair_df['type'].value_counts()
            n_intra = counts.get('intra', 0)
            n_inter = counts.get('inter', 0)
            self._log.info(f"   [Stats] Total: {total} | Intra: {n_intra} | Inter: {n_inter}")
        else:
            self._log.info(f"   [Stats] Total: {total} (No type info available)")