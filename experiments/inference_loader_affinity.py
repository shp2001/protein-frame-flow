import os
import numpy as np
import pandas as pd
import logging
import torch
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from data import utils as du


from openfold.data import data_transforms
from openfold.utils import rigid_utils
import json 
import re 

from data.motif_index import remask_antigen_mask, get_relpos_input, crop_antigen, load_general_mask_all, crop_general_protein, provide_anchor, get_ag_hotspot
from data import residue_constants as rc

from itertools import accumulate
import bisect

from data import featurizer
from torch.utils.data import SequentialSampler

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
    aatypes = processed_feats["aatype"].tolist()         # [L]
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
        None,
        )
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()
    res_mask[chain_feats['aatype'] == 20] = 0
    chain_idx = torch.tensor(processed_feats['chain_index'])
    residue_index = torch.tensor(processed_feats['residue_index'])

    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()

    return {
        'res_plddt': torch.tensor(res_plddt),
        'aatype': chain_feats['aatype'],
        'chain_index': chain_feats['chain_index'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'chain_idx': chain_idx,
        'residue_index': residue_index,
        'chain_seq_list': chain_seq_list,
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'],
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'],
    }


class BaseDataset(Dataset):
    def __init__(
            self,
            inf_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._inf_cfg = inf_cfg
        self._inference_cfg = inf_cfg.inference
        self.task = task        

        # affinity_csv 로드
        self.affinity_csv = pd.read_csv(self._inference_cfg.samples.csv_path_affinity)
        
        # metadata 로드
        self.meta_csv_path = self._inference_cfg.samples.csv_path
        self.chain_to_meta_row = self._load_metadata()
        
        # 필터링 (필요시)
        self._create_split(self.affinity_csv)
        
        self._cache = {}
        self._rng = np.random.default_rng(seed=123)

    def _load_metadata(self):
        """Training dataloader의 메타데이터 로딩 로직"""
        meta_df = pd.read_csv(self.meta_csv_path)
        
        if 'data_source' in meta_df.columns:
            return meta_df.set_index(['pdb_name', 'data_source']).to_dict(orient='index')
        else:
            return meta_df.set_index(['pdb_name']).to_dict(orient='index')

    @property
    def is_training(self):
        return self._is_training

    def __len__(self):
        return len(self.csv)
    
    def _create_split(self, data_csv):
        self.csv = data_csv
        self._log.info(f'Inference: {len(self.csv)} examples')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row, mut):
        path = csv_row['processed_path']
        processed_row = _process_csv_row(path, mut)

        mask_path = path.replace("meta", "mask_index").replace('.pkl', '.json')
        complex_id = os.path.basename(mask_path).replace('.json', '')

        if len(complex_id.split('_')) == 3:
            pdb_id, ligand_chains_str, receptor_chains_str = complex_id.split('_')
        elif len(complex_id.split('_')) == 4:
            pdb_id, h_str, l_str, ag_str = complex_id.split('_')
            ligand_chains_str = h_str + l_str
            receptor_chains_str = ag_str

        self.ligand_chains = list(ligand_chains_str)
        self.receptor_chains = list(receptor_chains_str)

        # TCR ligand-receptor mapping is changed 
        if "TCR" in csv_row['mode']:
            self.ligand_chains = list(receptor_chains_str)
            self.receptor_chains = list(ligand_chains_str)

        scaffold_mask = load_general_mask_all(
            mask_info_file=mask_path,
            residue_index=processed_row['residue_index'],
            chain_index=processed_row['chain_index'],
            threshold_length=25,
        )
        processed_row['loop_mask'] = torch.tensor(scaffold_mask)
        processed_row['mode'] = csv_row['mode']
        processed_row['raw_path'] = csv_row['raw_path'] 
        processed_row['selected_chains'] = self.ligand_chains

        return processed_row
        
    def __getitem__(self, row_idx):
        # Process data example.
        row = self.csv.iloc[row_idx]
        chains = str(row['mapped_chains']).split(';')
        mutations = str(row['mutation']).split(';')

        meta_key = chains[0].strip()
        mut = mutations[0].strip()
        has_source = 'Source Data Set' in row.index
        if has_source:
            source_dataset = row['Source Data Set']
            meta_key = (meta_key, source_dataset)
        meta_row = self.chain_to_meta_row[meta_key]
        feats = self.process_csv_row(meta_row, mut)
            
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
        loop_mask = provide_anchor(
            feats['loop_mask'],
            feats['res_mask'], 
            feats['chain_index'],
            feats['mode']
            ).to(torch.long)
        
        feats['loop_mask'] = loop_mask
        feats['diffuse_mask'] = diffuse_mask
        feats['mutation'] = mut

        # 3. Relative Position Encoding
        asym_id, entity_id, sym_id = get_relpos_input(feats['chain_seq_list'])
        feats['asym_id'] = torch.tensor(asym_id)
        feats['entity_id'] = torch.tensor(entity_id)
        feats['sym_id'] = torch.tensor(sym_id)

        feats['crop_idx'] = crop_general_protein(
            trans_1=feats['trans_1'],
            loop_mask=feats['loop_mask'],
            nan_mask=feats['res_mask'],
            max_len=400,
        )

        # affinity 정보 추가
        feats['affinity_kd'] = row['Affinity_Kd [nM]']
        feats['csv_idx'] = torch.tensor(row_idx, dtype=torch.long)
        feats['processed_path'] = meta_row['processed_path']
        feats['mutation'] = mut
        if has_source:
            feats['data_source'] = row['Source Data Set']
        return feats

def collate_single_side(batch_feats):
    """
    [Residue Level Processing]
    단일 feature 리스트를 받아 Crop -> Stack -> Center -> Hotspot 과정을 수행합니다.
    """
    cropped_batch = []
    
    # Crop 대상에서 제외할 키 리스트
    not_crop_key = [
        'crop_idx', 'scaffold_idx', 'chain_seq_list', 'masked_chain', 
        'first_chain_len', 'raw_path', 'mode', 'mutation', 'processed_path',
        'affinity_kd', 'csv_idx', 'data_source'
    ]

    for feat in batch_feats:
        cropped_feat = {}
        crop_idx = feat['crop_idx']
        seq_len = feat['residue_index'].shape[0]

        for key, value in feat.items():
            # 1. 크롭 제외 대상이거나 텐서가 아닌 경우 그대로 유지
            if key in not_crop_key or not isinstance(value, torch.Tensor):
                cropped_feat[key] = value
                continue
            
            # 2. 텐서인 경우 차원(ndim)에 따른 처리
            # 0차원 텐서(스칼라)는 shape[0] 접근 시 에러가 나므로 ndim >= 1 체크 필수
            if value.ndim >= 2 and value.shape[0] == seq_len and value.shape[1] == seq_len:
                # 2D Pairwise Feature Cropping
                cropped_feat[key] = value[crop_idx][:, crop_idx]
            elif value.ndim >= 1 and value.shape[0] == seq_len:
                # 1D Sequence Feature Cropping
                cropped_feat[key] = value[crop_idx]
            else:
                # 스칼라 텐서이거나, 시퀀스 길이와 무관한 텐서
                cropped_feat[key] = value
                
        cropped_batch.append(cropped_feat)

    # 3. Stack Features (배치 단위로 묶기)
    first_keys = cropped_batch[0].keys()
    collated_batch = {}
    
    for key in first_keys:
        val = cropped_batch[0][key]
        if isinstance(val, torch.Tensor):
            # 0차원 텐서들의 경우 stack하면 [Batch] 크기의 1차원 텐서가 됨
            collated_batch[key] = torch.stack([d[key] for d in cropped_batch], dim=0)
        else:
            collated_batch[key] = [d[key] for d in cropped_batch]

    # 4. Metadata 처리 (배치의 마지막 샘플 기준 정보 보존)
    # 기존 코드의 로직을 유지하되 안전하게 할당
    last_feat = batch_feats[-1]
    collated_batch['raw_path'] = last_feat.get('raw_path')
    collated_batch['mode'] = last_feat.get('mode')
    collated_batch['mutation'] = last_feat.get('mutation')
    
    # 5. Ag Hotspot 계산
    collated_batch['ag_hotspot'] = get_ag_hotspot(
        collated_batch['pseudo_beta'],
        collated_batch['loop_mask'],
        collated_batch.get('diffuse_mask', None),
        threshold=8,
        masking_ratio=1,
        false_hotspot_ratio=0,
        noise_range=4
    )

    # 6. Center based on motif locations (중심점 보정)
    motif_mask = 1 - collated_batch['loop_mask']
    # motif_mask: [B, L], trans_1: [B, L, 3]
    motif_1 = collated_batch['trans_1'] * motif_mask[..., None]
    motif_sum = torch.sum(motif_mask, dim=1) + 1e-8
    motif_com = torch.sum(motif_1, dim=1) / motif_sum[..., None] # [B, 3]

    # 좌표 업데이트
    collated_batch["trans_1"] = collated_batch['trans_1'] - motif_com[:, None, :]
    collated_batch['pseudo_beta'] = collated_batch['pseudo_beta'] - motif_com[:, None, :]
    
    if 'atom14_gt_positions' in collated_batch:
        collated_batch['atom14_gt_positions'] = collated_batch['atom14_gt_positions'] - motif_com[:, None, None, :]
    
    # alt_gt_positions는 존재 여부 확인 후 처리
    if 'atom14_alt_gt_positions' in collated_batch:
        collated_batch['atom14_alt_gt_positions'] = collated_batch['atom14_alt_gt_positions'] - motif_com[:, None, None, :]
    
    # 7. Edge mask (Residue level)
    # res_mask: [B, L] -> edge_mask: [B, L, L]
    collated_batch["edge_mask"] = collated_batch['res_mask'][:, None] * collated_batch['res_mask'][:, :, None]
    
    return collated_batch

def post_process_atom_features(batch_dict):
    """
    [Atom Level Processing]
    Split이 완료된 이후 호출되어야 합니다.
    1. Featurizer: 현재 데이터(Complex/Ligand/Receptor)의 aatype에 맞는 Reference Feature 생성.
    2. Atom Mask & Flatten: Atom Level Mask를 생성하고 Flatten 수행.
    """
    
    # 1. Featurizer
    # batch_dict['aatype']은 Split 여부에 따라 전체 시퀀스일 수도, Ligand/Receptor 부분일 수도 있음.
    ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, atom_to_tokatom_idx, ref_pos = \
        featurizer.get_ref_basic_feature(
            batch_dict['aatype'], 
            batch_dict['atom14_gt_exists'], 
            batch_dict['residue_index']
        )
    
    batch_dict['ref_feature_dict'] = {
        'ref_space_uid': ref_space_uid,
        'atom_to_token_idx': atom_to_token_idx,
        'atom_to_tokatom_idx': atom_to_tokatom_idx,
        'ref_pos': ref_pos,
        'ref_element': ref_element,
        'ref_charge': ref_charge,
        'ref_atom_name_chars': ref_atom_name_chars,
    }

    # 2. Create atom masks (In-place 연산 제거 버전)
    atom_diffuse_mask = batch_dict['atom14_gt_exists'].clone()
    diffuse_val = atom_diffuse_mask[..., :3] * batch_dict["diffuse_mask"].unsqueeze(-1)
    atom_diffuse_mask[..., :3] = diffuse_val
    batch_dict['atom_diffuse_mask'] = du.atom_flatten(atom_diffuse_mask, batch_dict['atom14_gt_exists'])
    batch_dict['r_1'] = du.atom_flatten(batch_dict['atom14_gt_positions'], batch_dict['atom14_gt_exists'])
    
    return batch_dict

def split_complex_data(complex_data, ligand_mask, target_val):
    """
    complex -> ligand/receptor 분리 (Indices Slicing)
    """
    mask_0 = ligand_mask[0]
    indices = torch.nonzero(mask_0 == target_val, as_tuple=False).squeeze(-1)
    seq_len = mask_0.shape[0]

    def recursive_slice(data):
        if isinstance(data, dict):
            return {k: recursive_slice(v) for k, v in data.items()}
        
        elif isinstance(data, torch.Tensor):
            # 2D Pairwise Feature [B, L, L, ...]
            if data.ndim >= 3 and data.shape[1] == seq_len and data.shape[2] == seq_len:
                return data[:, indices][:, :, indices]
            # 1D Sequence Feature [B, L, ...]
            elif data.ndim >= 2 and data.shape[1] == seq_len:
                return data[:, indices]
            else:
                return data
        else:
            return data

    return recursive_slice(complex_data)

def collate_fn(batch):
    """
    batch: List of dicts. 
    """
    use_unbound = True
    complex_data = collate_single_side(batch)
    
    # 2. Unbound 처리가 필요한 경우 (Ligand/Receptor 분리)
    if use_unbound:
        # Ligand Mask 추출 (collate_single_side에서 처리되지 않았을 경우를 대비)
        if 'ligand_mask' in complex_data:
            ligand_mask = complex_data['ligand_mask']
        else:
            ligand_mask = torch.stack([f['ligand_mask'] for f in batch], dim=0)

        # [Ligand] 분리 및 원자 레벨 처리
        ligand_data = split_complex_data(complex_data, ligand_mask, target_val=1)
        post_process_atom_features(ligand_data)
        
        # [Receptor] 분리 및 원자 레벨 처리
        receptor_data = split_complex_data(complex_data, ligand_mask, target_val=0)
        post_process_atom_features(receptor_data)

        # [Complex] 원본 원자 레벨 처리
        post_process_atom_features(complex_data)

        # 결과 구조화
        batch_output = {
            "complex": complex_data,
            "ligand": ligand_data,
            "receptor": receptor_data
        }
    else:
        # Unbound 미사용 시 Complex만 처리
        post_process_atom_features(complex_data)
        batch_output = {"complex": complex_data}

    return batch_output

def predict_dataloader(dataset,
                       loader_cfg):
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        sampler=SequentialSampler(dataset),
        prefetch_factor=None if loader_cfg.num_workers == 0 else loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if loader_cfg.num_workers > 0 else False,
        collate_fn=collate_fn
    )