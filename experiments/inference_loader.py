import abc
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
import random
import re
import os
import copy
from data.motif_index import get_relpos_input, crop_antigen, provide_anchor, get_ag_hotspot, crop_general_affinity
from data import residue_constants as rc
from itertools import accumulate
import bisect
from data import featurizer
from torch.utils.data import SequentialSampler

def _process_mutations(processed_file_path, mut, scaffold_idx):
    """Training dataloader와 동일한 mutation 처리 로직"""
    processed_feats = du.read_pkl(processed_file_path)
    
    # [Safety] scaffold_idx 원본 보존을 위해 Deep Copy 수행
    updated_scaffold_idx = copy.deepcopy(scaffold_idx)
    
    # 텐서 변환
    for k, v in processed_feats.items():
        if not isinstance(v, torch.Tensor):
            processed_feats[k] = torch.tensor(v)

    if mut == 'No_Mutation':
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
            for key in processed_feats.keys():
                feat = processed_feats[key]
                processed_feats[key] = torch.cat([feat[:idx], feat[idx+1:]], dim=0)
            
            mask_update = (processed_feats['chain_index'] == target_chain_idx) & \
                        (processed_feats['residue_index'] > target_res_id)
            processed_feats['residue_index'][mask_update] -= 1

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

            mask_update = (processed_feats['chain_index'] == target_chain_idx) & \
                        (processed_feats['residue_index'] > target_res_id)
            processed_feats['residue_index'][mask_update] += 1
            
            for key in processed_feats.keys():
                curr_data = processed_feats[key]
                to_insert = new_row[key]
                processed_feats[key] = torch.cat([
                    curr_data[:insert_pos],
                    to_insert,
                    curr_data[insert_pos:]
                ], dim=0)

            for region_key in updated_scaffold_idx:
                if updated_scaffold_idx[region_key] >= insert_pos:
                    updated_scaffold_idx[region_key] += 1

        # --- Substitution ---
        else:
            mutated_aa_char = type_str
            mutated_aa_idx = rc.restype_order.get(mutated_aa_char, 20)
            processed_feats['aatype'][idx] = mutated_aa_idx
            processed_feats['atom_mask'][idx] = torch.tensor(rc.STANDARD_ATOM_MASK[mutated_aa_idx])

    return processed_feats, updated_scaffold_idx

def _process_csv_row(processed_file_path, mut, scaffold_idx):
    """Training dataloader와 동일한 CSV row 처리 로직"""
    processed_feats = du.read_pkl(processed_file_path)
    if mut != "No_Mutation":
        processed_feats, scaffold_idx = _process_mutations(processed_file_path, mut, scaffold_idx)
    processed_feats = du.parse_chain_feats(processed_feats)

    # make chain sequence list (for the multimer relpos embedding)
    int_to_aa = {i: restype for restype, i in rc.restype_order_with_x.items()}
    aatypes = processed_feats["aatype"].tolist()
    chain_indices = processed_feats["chain_index"].tolist()

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
        'scaffold_idx': scaffold_idx,
        'chain_seq_list': chain_seq_list,
        'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        'alt_torsion_angles_sin_cos': chain_feats['alt_torsion_angles_sin_cos'],
        'torsion_angles_mask': chain_feats['torsion_angles_mask'],
        'chi_angles_sin_cos': chain_feats['chi_angles_sin_cos'],
        'chi_mask': chain_feats['chi_mask'],
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'],
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'],
        'atom14_alt_gt_positions': chain_feats['atom14_alt_gt_positions'],
        'atom14_alt_gt_exists': chain_feats['atom14_alt_gt_exists'],
        'atom14_atom_is_ambiguous': chain_feats['atom14_atom_is_ambiguous'],
        'backbone_rigid_mask': chain_feats['backbone_rigid_mask'],
        'rigidgroups_gt_frames': chain_feats['rigidgroups_gt_frames'],
        'rigidgroups_gt_exists': chain_feats['rigidgroups_gt_exists'],
        'rigidgroups_alt_gt_frames': chain_feats['rigidgroups_alt_gt_frames'],
    }


class BaseDataset(Dataset):
    def __init__(self, inf_cfg, is_training, task):
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
        """Training dataloader의 process_csv_row 로직 사용"""
        # metadata에서 정보 가져오기
        chains = str(csv_row['mapped_chains']).split(';')
        mutations = str(csv_row['mutation']).split(';')
        
        # 랜덤하게 하나 선택 (또는 첫 번째 선택)
        part_idx = random.randint(0, len(chains) - 1)
        
        meta_key = chains[part_idx].strip()
        selected_mutation = mutations[part_idx].strip()
        
        has_source = 'Source Data Set' in csv_row.index
        if has_source:
            source_dataset = csv_row['Source Data Set']
            meta_key = (meta_key, source_dataset)
        
        selected_meta_row = self.chain_to_meta_row[meta_key]
        path = selected_meta_row['processed_path']
        
        # scaffold_idx 구성
        scaffold_idx = {}
        
        # CDR 정보가 있는 경우 (ab mode)
        if "h1_start" in selected_meta_row:
            cdr_types = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']
            for cdr in cdr_types:
                scaffold_idx[f'{cdr}_start'] = int(selected_meta_row[f'{cdr}_start'])
                scaffold_idx[f'{cdr}_end'] = int(selected_meta_row[f'{cdr}_end'])
        else:
            # affinity mode
            mask_path = path.replace("meta", "mask_index").replace('.pkl', '.json')
            complex_id = os.path.basename(mask_path).replace('.json', '')
            pdb_id, ligand_chains_str, receptor_chains_str = complex_id.split('_')
            self.ligand_chains = list(ligand_chains_str)
            self.receptor_chains = list(receptor_chains_str)

            with open(mask_path, 'r') as f:
                mask_info = json.load(f)
            
            if "TCR" in selected_meta_row.get('mode', ''):
                self.ligand_chains = list(receptor_chains_str)
                self.receptor_chains = list(ligand_chains_str)

            for chain_id in self.ligand_chains:
                if chain_id not in mask_info:
                    continue
                for idx, block in enumerate(mask_info[chain_id]):
                    scaffold_idx[f"{chain_id}_{idx}_start"] = block[0]
                    scaffold_idx[f"{chain_id}_{idx}_end"] = block[-1]

            if selected_mutation != "No_Mutation":
                mut_parts = selected_mutation.split('_')
                for part in mut_parts:
                    m_chain = part[1]
                    m_residue = int(part[2:-1])

                    if m_chain in self.receptor_chains and m_chain in mask_info:
                        for idx, block in enumerate(mask_info[m_chain]):
                            if m_residue in block:
                                scaffold_idx[f"{m_chain}_{idx}_start"] = block[0]
                                scaffold_idx[f"{m_chain}_{idx}_end"] = block[-1]
                                break

        # 캐시 사용
        use_cache = True
        cache_key = (path, selected_mutation)
        if use_cache and cache_key in self._cache:
            processed_row = self._cache[cache_key]
        else:
            processed_row = _process_csv_row(path, selected_mutation, scaffold_idx)
            processed_row['mode'] = selected_meta_row.get('mode', 'affinity')
            processed_row['raw_path'] = selected_meta_row.get('raw_path', '')
            if use_cache:
                self._cache[cache_key] = processed_row
        
        return processed_row

    def _sample_scaffold_mask(self, batch):
        """Training dataloader의 scaffold_mask 생성 로직"""
        aatype = batch['aatype']
        mode = batch['mode']
        num_res = aatype.shape[0]
        scaffold_idx = batch['scaffold_idx']
        scaffold_mask = torch.zeros(num_res)

        if mode == 'ab':
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(6):
                scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1

        if mode == 'nanobody':
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(3):
                scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1

        if 'affinity' in mode:
            interface_points = []
            for key, target_resid in scaffold_idx.items():
                chain_str = key.split('_')[0]
                ci_int = du.chain_str_to_int(chain_str)

                found_idx = -1
                for i, (b_chain, b_res) in enumerate(zip(batch['chain_index'], batch['residue_index'])):
                    if b_chain == ci_int and b_res == target_resid:
                        found_idx = i
                        break
                
                if found_idx != -1:
                    interface_points.append(found_idx)

            interface_points = sorted(interface_points)
            for i in range(len(interface_points)//2):
                scaffold_mask[interface_points[2*i]:interface_points[2*i+1]+1] = 1
                if interface_points[2*i] == interface_points[2*i+1]:
                    scaffold_mask[interface_points[2*i]-1:interface_points[2*i+1]+2] = 1

        return scaffold_mask * batch['res_mask']

    def setup_inpainting(self, feats):
        """Training dataloader의 inpainting 설정 로직"""
        loop_mask = self._sample_scaffold_mask(feats)
        feats['loop_mask'] = loop_mask

    def __getitem__(self, row_idx):
        csv_row = self.csv.iloc[row_idx]
        
        # mutation 정보 가져오기
        mutations = str(csv_row['mutation']).split(';')
        chains = str(csv_row['mapped_chains']).split(';')
        
        # 랜덤하게 하나 선택 (또는 첫 번째 선택)
        part_idx = random.randint(0, len(chains) - 1)
        selected_mutation = mutations[part_idx].strip()
        
        # metadata 정보 가져오기
        meta_key = chains[part_idx].strip()
        has_source = 'Source Data Set' in csv_row.index
        if has_source:
            source_dataset = csv_row['Source Data Set']
            meta_key = (meta_key, source_dataset)
        
        selected_meta_row = self.chain_to_meta_row[meta_key]
        
        # 데이터 처리
        feats = self.process_csv_row(csv_row, selected_mutation)
        
        # rigid transforms
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(feats['rigidgroups_gt_frames'])[:, 0]
        feats['rotmats_1'] = torch.tensor(rigids_1.get_rots().get_rot_mats(), device=rigids_1.device)
        feats['trans_1'] = torch.tensor(rigids_1.get_trans(), device=rigids_1.device)

        # inpainting 설정
        self.setup_inpainting(feats)
        
        # anchor 제공
        feats['loop_mask'] = provide_anchor(
            feats['loop_mask'], 
            feats['res_mask'], 
            feats['chain_index'],
            feats['mode']
        ).to(torch.long)
        
        # diffuse_mask 생성
        chain_len_list = [len(seq) for seq in feats['chain_seq_list']]
        if len(chain_len_list) == 1:
            diffuse_mask = feats['loop_mask']
        elif 'affinity' not in feats['mode']:
            asym_id = []
            for chain_idx, chain_len in enumerate(chain_len_list):
                for _ in range(chain_len):
                    asym_id.append(chain_idx)
            asym_id = torch.tensor(asym_id, device=feats['loop_mask'].device)
            masked_chain = asym_id[feats['loop_mask'] == 1].unique()
            diffuse_mask = torch.isin(asym_id, masked_chain).to(torch.long)
        elif 'affinity' in feats['mode']:
            ligand_chain_indices = [du.chain_str_to_int(c) for c in self.ligand_chains]
            ligand_chain_tensor = torch.tensor(ligand_chain_indices, device=feats['loop_mask'].device)
            chain_index = feats['chain_index']
            ligand_mask = torch.isin(chain_index, ligand_chain_tensor).to(torch.long)
            diffuse_mask = torch.max(ligand_mask, feats['loop_mask'])
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
                f"diffuse_mask is all zero for sample at index {row_idx}. "
                "Check if ligand_chains or loop_mask (CDR/Interface) are correctly defined."
            )
        
        # crop 처리
        if feats['mode'] in ['ab', 'nanobody']:
            feats['crop_idx'] = crop_antigen(
                feats['trans_1'],
                cdr_mask=feats['loop_mask'],
                nan_mask=feats['res_mask'],
                max_len=self._inference_cfg.ab_max_num_res,
                seq_list=feats['chain_seq_list'],
                crop_ab=self._inference_cfg.crop_ab,
                mode=feats['mode'],
            )
        else:
            # affinity mode - mask_path 생성
            mask_path = selected_meta_row['processed_path'].replace("meta", "mask_index").replace('.pkl', '.json')
            with open(mask_path, 'r') as f:
                mask_info = json.load(f)
            
            if torch.sum(feats['loop_mask']) == 0:
                print("mask_path", mask_path)
            
            feats['crop_idx'] = crop_general_affinity(
                trans_1=feats['trans_1'],
                loop_mask=feats['loop_mask'],
                nan_mask=feats['res_mask'],
                max_len=self._inference_cfg.general_max_num_res,
                mask_info=mask_info,
                residue_index=feats['residue_index'],
                chain_index=feats['chain_index'],
                max_res_num_interface=self._inference_cfg.max_mask_residues
            )
        
        # affinity 정보 추가
        feats['affinity_kd'] = csv_row['Affinity_Kd [nM]']
        feats['csv_idx'] = torch.tensor(row_idx, dtype=torch.long)
        feats['processed_path'] = selected_meta_row['processed_path']
        feats['mutation'] = selected_mutation
        if has_source:
            feats['data_source'] = csv_row['Source Data Set']
        return feats


def collate_fn(batch):
    """Training dataloader의 collate_fn과 유사하게 수정"""
    cropped_batch = []
    
    for feat in batch:
        mode = feat['mode']
        cropped_feat = {}
        
        # crop_idx가 이미 __getitem__에서 생성되었으므로 그대로 사용
        crop_idx = feat['crop_idx']
        
        not_crop_key = [
            'crop_idx', 'scaffold_idx', 'chain_seq_list', 'csv_idx', 
            'raw_path', 'mode', 'affinity_kd', 'processed_path', 'mutation', 'data_source'
        ]
        
        for key in feat.keys():
            if key not in not_crop_key:
                cropped_feat[key] = feat[key][crop_idx]
            elif key == 'chain_seq_list':
                lengths = [len(s) for s in feat[key]]
                start_positions = list(accumulate([0] + lengths))
                merged = "".join(feat[key])
                cropped_seq_list = [[] for _ in range(len(feat[key]))]
                for idx in crop_idx:
                    chain_idx = bisect.bisect_right(start_positions, idx) - 1
                    cropped_seq_list[chain_idx].append(merged[idx])
                cropped_feat[key] = ["".join(chain_seq) for chain_seq in cropped_seq_list]
        
        # relpos 생성
        asym_id, entity_id, sym_id = get_relpos_input(cropped_feat['chain_seq_list'])
        cropped_feat['asym_id'] = torch.tensor(asym_id)
        cropped_feat['entity_id'] = torch.tensor(entity_id)
        cropped_feat['sym_id'] = torch.tensor(sym_id)
        
        cropped_feat['csv_idx'] = feat['csv_idx']
        cropped_feat['crop_idx'] = torch.tensor(crop_idx)
        cropped_feat['affinity_kd'] = torch.tensor(feat['affinity_kd'])
        
        del cropped_feat['chain_seq_list']
        cropped_batch.append(cropped_feat)
    
    # 배치로 합치기
    cropped_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}
    for key in cropped_batch.keys():
        if key not in ['mode', 'raw_path']:  # 문자열은 스택하지 않음
            cropped_batch[key] = torch.stack(cropped_batch[key], dim=0)
    
    # ref feature 생성
    ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, atom_to_tokatom_idx, ref_pos = featurizer.get_ref_basic_feature(
        cropped_batch['aatype'], 
        cropped_batch['atom14_gt_exists'], 
        cropped_batch['residue_index']
    )
    cropped_batch['ref_feature_dict'] = {
        'ref_space_uid': ref_space_uid,
        'ref_element': ref_element,
        'ref_charge': ref_charge,
        'ref_atom_name_chars': ref_atom_name_chars,
        'atom_to_token_idx': atom_to_token_idx,
        'atom_to_tokatom_idx': atom_to_tokatom_idx,
        'ref_pos': ref_pos,
    }
    
    # ag_hotspot
    cropped_batch['ag_hotspot'] = get_ag_hotspot(
        cropped_batch['pseudo_beta'],
        cropped_batch['loop_mask'],
        cropped_batch['diffuse_mask'],
        threshold=8
    )
    
    # Centering
    motif_mask = 1 - cropped_batch['loop_mask']
    motif_1 = cropped_batch['trans_1'] * motif_mask[..., None]
    motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask, dim=1) + 1)[..., None]
    cropped_batch["trans_1"] = cropped_batch['trans_1'] - motif_com[:, None, :]
    cropped_batch['atom14_gt_positions'] = cropped_batch['atom14_gt_positions'] - motif_com[:, None, None, :]
    cropped_batch['pseudo_beta'] = cropped_batch['pseudo_beta'] - motif_com[:, None, :]
    
    # atom diffuse_mask
    atom_diffuse_mask = cropped_batch['atom14_gt_exists'].clone()
    atom_diffuse_mask[..., :3] *= cropped_batch["diffuse_mask"].unsqueeze(-1)
    cropped_batch['atom_diffuse_mask'] = du.atom_flatten(atom_diffuse_mask, cropped_batch['atom14_gt_exists'])
    cropped_batch['r_1'] = du.atom_flatten(cropped_batch['atom14_gt_positions'], cropped_batch['atom14_gt_exists'])
    
    # edge mask
    cropped_batch["edge_mask"] = cropped_batch['res_mask'][:, None] * cropped_batch['res_mask'][:, :, None]
    cropped_batch["mutation"] = feat['mutation']
    cropped_batch['processed_path'] = feat['processed_path']
    if 'data_soruce' in feat:
        cropped_batch['data_source'] = feat['data_source']
    cropped_batch['mode'] = feat['mode']
    return cropped_batch


def predict_dataloader(dataset, loader_cfg):
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