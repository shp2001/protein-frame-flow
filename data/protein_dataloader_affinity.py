"""Protein data loader."""
import torch
import pandas as pd
import logging
import random
import numpy as np
from itertools import accumulate
import bisect

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.motif_index import get_ag_hotspot
from data import featurizer
from data import utils as du 

class DynamicDistributedSampler(DistributedSampler):
    """
    Dataset의 길이가 epoch마다 동적으로 변하는 경우를 위한 DistributedSampler.
    num_samples와 total_size를 property로 만들어 매번 동적으로 계산합니다.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        # 부모 클래스 초기화 (num_samples, total_size는 property로 대체)
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
    
    @property
    def num_samples(self):
        """매번 동적으로 계산"""
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            return len(self.dataset) // self.num_replicas
        else:
            return (len(self.dataset) + self.num_replicas - 1) // self.num_replicas
    
    @property
    def total_size(self):
        """매번 동적으로 계산"""
        return self.num_samples * self.num_replicas
    
    def __iter__(self):
        # Shuffle 또는 순차 인덱스 생성
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Padding 또는 Drop
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * ((padding_size // len(indices)) + 1))[:padding_size]
        else:
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

class ProteinData(LightningDataModule):

    def __init__(self, *, data_cfg, train_dataset, valid_dataset, predict_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.hotspot_cfg = data_cfg.hotspot
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset
        self._current_epoch = 0  

    def set_current_epoch(self, epoch):
        self._current_epoch = epoch
        self._train_dataset.set_current_epoch(epoch)

    def _collate_single_side(self, batch_feats):
        """
        [Residue Level Processing]
        단일 feature 리스트를 받아 Crop -> Stack -> Center -> Hotspot 과정을 수행합니다.
        
        Note:
        - Featurizer(Ref features)와 Atom Flatten(r_1, masks) 로직은 여기서 수행하지 않습니다.
        - 이는 Split 이후 정확한 aatype 참조와 Mask 처리를 위해 _post_process_atom_features로 이동되었습니다.
        """
        cropped_batch = []
        for feat in batch_feats:
            cropped_feat = {}
            not_crop_key = ['crop_idx', 'scaffold_idx', 'chain_seq_list', 'masked_chain', 
                            'first_chain_len', 'raw_path', 'mode', 'mutation']
            
            crop_idx = feat['crop_idx']
            seq_len = feat['residue_index'].shape[0] 

            # 1. Feature Cropping
            for key, value in feat.items():
                if key in not_crop_key:
                    continue
                
                if isinstance(value, torch.Tensor):
                    # 2D Pairwise Feature Cropping
                    if value.ndim >= 2 and value.shape[0] == seq_len and value.shape[1] == seq_len:
                        cropped_feat[key] = value[crop_idx][:, crop_idx]
                    # 1D Sequence Feature Cropping
                    elif value.shape[0] == seq_len:
                        cropped_feat[key] = value[crop_idx]
                    else:
                        cropped_feat[key] = value
                else:
                    cropped_feat[key] = value
            cropped_batch.append(cropped_feat)

        # 4. Stack Features
        first_keys = cropped_batch[0].keys()
        collated_batch = {}
        
        for key in first_keys:
            val = cropped_batch[0][key]
            if isinstance(val, torch.Tensor):
                collated_batch[key] = torch.stack([d[key] for d in cropped_batch], dim=0)
            else:
                collated_batch[key] = [d[key] for d in cropped_batch]

        # --- Metadata & Hotspot ---
        collated_batch['raw_path'] = feat['raw_path']
        collated_batch['mode'] = feat['mode']
        collated_batch['mutation'] = feat['mutation']

        # 6. Ag Hotspot
        collated_batch['ag_hotspot'] = get_ag_hotspot(
            collated_batch['pseudo_beta'],
            collated_batch['loop_mask'],
            collated_batch.get('diffuse_mask', None),
            threshold=self.hotspot_cfg.threshold,
            masking_ratio=self.hotspot_cfg.hotspot_noise.masking_ratio,
            false_hotspot_ratio=self.hotspot_cfg.hotspot_noise.false_hotspot_ratio,
            noise_range=self.hotspot_cfg.hotspot_noise.noise_range
        )

        # 7. Center based on motif locations
        motif_mask = 1 - collated_batch['loop_mask']
        motif_1 = collated_batch['trans_1'] * motif_mask[..., None]
        motif_sum = torch.sum(motif_mask, dim=1) + 1e-8
        motif_com = torch.sum(motif_1, dim=1) / motif_sum[..., None]

        collated_batch["trans_1"] = collated_batch['trans_1'] - motif_com[:, None, :]
        collated_batch['atom14_gt_positions'] = collated_batch['atom14_gt_positions'] - motif_com[:, None, None, :]
        collated_batch['atom14_alt_gt_positions'] = collated_batch['atom14_alt_gt_positions'] - motif_com[:, None, None, :]
        collated_batch['pseudo_beta'] = collated_batch['pseudo_beta'] - motif_com[:, None, :]
        
        # 9. Edge mask (Residue level)
        collated_batch["edge_mask"] = collated_batch['res_mask'][:, None] * collated_batch['res_mask'][:, :, None]
        
        return collated_batch

    def _post_process_atom_features(self, batch_dict):
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

    def _split_complex_data(self, complex_data, ligand_mask, target_val):
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

    def collate_fn(self, batch):
        """
        batch: List of dicts. 
        Structure: batch -> batch_0 / batch_1 -> complex / ligand / receptor
        """
        batch_output = {}
        use_unbound = self.data_cfg.use_unbound

        # ==========================================
        # 1. Batch 0 Processing
        # ==========================================
        feats_0 = [item["batch_0"] for item in batch]
        
        # 1-1. Complex 생성 (Residue Level)
        complex_data_0 = self._collate_single_side(feats_0)
        
        if use_unbound:
            # Ligand Mask 추출
            if 'ligand_mask' in complex_data_0:
                ligand_mask = complex_data_0['ligand_mask']
            else:
                # 만약 collate 과정에서 제외되었다면 원본에서 복구
                ligand_mask = torch.stack([f['ligand_mask'] for f in feats_0], dim=0)

            # 1-2. Split & Featurize & Flatten
            
            # [Ligand]
            ligand_data_0 = self._split_complex_data(complex_data_0, ligand_mask, target_val=1)
            self._post_process_atom_features(ligand_data_0)
            
            # [Receptor]
            receptor_data_0 = self._split_complex_data(complex_data_0, ligand_mask, target_val=0)
            self._post_process_atom_features(receptor_data_0)

            # [Complex] (Split 이후 원본 처리)
            self._post_process_atom_features(complex_data_0)

            # 구조화
            batch_output['batch_0'] = {
                "complex": complex_data_0,
                "ligand": ligand_data_0,
                "receptor": receptor_data_0
            }
        else:
            # Unbound 미사용 시 Complex만 처리
            self._post_process_atom_features(complex_data_0)
            batch_output['batch_0'] = {"complex": complex_data_0}


        # ==========================================
        # 2. Batch 1 Processing
        # ==========================================
        # Batch 1 데이터가 Batch 0와 동일하게 flat dict list로 들어온다고 가정
        feats_1 = [item["batch_1"] for item in batch]
        complex_data_1 = self._collate_single_side(feats_1)
        
        if use_unbound:
            if 'ligand_mask' in complex_data_1:
                ligand_mask_1 = complex_data_1['ligand_mask']
            else:
                ligand_mask_1 = torch.stack([f['ligand_mask'] for f in feats_1], dim=0)

            # [Ligand]
            ligand_data_1 = self._split_complex_data(complex_data_1, ligand_mask_1, target_val=1)
            self._post_process_atom_features(ligand_data_1)
            
            # [Receptor]
            receptor_data_1 = self._split_complex_data(complex_data_1, ligand_mask_1, target_val=0)
            self._post_process_atom_features(receptor_data_1)

            # [Complex]
            self._post_process_atom_features(complex_data_1)

            batch_output['batch_1'] = {
                "complex": complex_data_1,
                "ligand": ligand_data_1,
                "receptor": receptor_data_1
            }
        else:
            self._post_process_atom_features(complex_data_1)
            batch_output['batch_1'] = {"complex": complex_data_1}
        

        # ==========================================
        # 3. Labels & Metadata
        # ==========================================
        labels = [item['label'] for item in batch]
        batch_output['label'] = torch.tensor(labels, dtype=torch.long)
        
        if 'kd1' in batch[0]:
            kd1s = [item['kd1'] for item in batch]
            batch_output['kd1'] = torch.tensor(kd1s, dtype=torch.float)
        
        if 'kd2' in batch[0]:
            kd2s = [item['kd2'] for item in batch]
            batch_output['kd2'] = torch.tensor(kd2s, dtype=torch.float)

        return batch_output
        
    def train_dataloader(self, rank=None, num_replicas=None):
        return DataLoader(
            self._train_dataset,
            sampler=DynamicDistributedSampler(
                self._train_dataset,
                shuffle=True,
                drop_last=True
            ),
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if self.loader_cfg.num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            batch_size=1,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if self.loader_cfg.num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )