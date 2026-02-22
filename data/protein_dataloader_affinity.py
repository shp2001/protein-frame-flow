"""Protein data loader."""
import torch
import pandas as pd
import logging
import random
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist

from data.motif_index import get_relpos_input, get_ag_hotspot
from data import featurizer
from data import utils as du 

import analysis.utils as au 
from itertools import accumulate
import bisect

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
        feats_paired의 0번 혹은 1번 키에 해당하는 feature list를 처리하는 함수.
        기존 collate_fn 로직을 이곳으로 옮김.
        """
        cropped_batch = []
        for feat in batch_feats:
            cropped_feat = {}
            not_crop_key = ['crop_idx', 'scaffold_idx', 'chain_seq_list', 'masked_chain', 'first_chain_len', 'raw_path', 'mode']
            for key in feat.keys():
                if key not in not_crop_key:
                    cropped_feat[key] = feat[key][feat['crop_idx']]

                if key == 'chain_seq_list':
                    lengths = [len(s) for s in feat[key]]
                    start_positions = list(accumulate([0] + lengths))
                    merged = "".join(feat[key])
                    cropped_seq_list = [[] for _ in range(len(feat[key]))]

                    for idx in feat['crop_idx']:
                        chain_idx = bisect.bisect_right(start_positions, idx) - 1
                        cropped_seq_list[chain_idx].append(merged[idx])

                    cropped_feat[key] = ["".join(chain_seq) for chain_seq in cropped_seq_list]

            asym_id, entity_id, sym_id = get_relpos_input(cropped_feat['chain_seq_list'])
            cropped_feat['asym_id'] = torch.tensor(asym_id)
            cropped_feat['entity_id'] = torch.tensor(entity_id)
            cropped_feat['sym_id'] = torch.tensor(sym_id)
            cropped_feat['crop_idx'] = torch.tensor(feat['crop_idx'])
            del cropped_feat['chain_seq_list']
            cropped_batch.append(cropped_feat)

        # 2. Stack Features (List of dicts -> Dict of tensors)
        # 모든 샘플의 길이가 같다고 가정 (Batcher에서 길이별로 묶었으므로)
        collated_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}   

        for key in collated_batch.keys():
            collated_batch[key] = torch.stack(collated_batch[key], dim=0)   


        # 메타 데이터는 첫 번째 샘플 기준으로 (mode 등은 배치 내에서 동일하거나 리스트로 관리 필요하지만 기존 로직 따름)
        # mode나 raw_path가 배치 내에서 다를 수 있다면 리스트로 유지하는 것이 좋으나, 
        # 기존 코드 호환성을 위해 tensor 변환이 안되는 항목은 리스트 혹은 첫번째 값 사용
        collated_batch['mode'] = batch_feats[0]['mode'] 
        collated_batch['raw_path'] = batch_feats[0]['raw_path']

        # 3. Featurizer & Masks
        ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, atom_to_tokatom_idx, ref_pos = \
            featurizer.get_ref_basic_feature(collated_batch['aatype'], collated_batch['atom14_gt_exists'], collated_batch['residue_index'])
        
        collated_batch['ref_feature_dict'] = {
            'ref_space_uid': ref_space_uid,
            'atom_to_token_idx': atom_to_token_idx,
            'atom_to_tokatom_idx': atom_to_tokatom_idx,
            'ref_pos': ref_pos,
            'ref_element': ref_element,
            'ref_charge': ref_charge,
            'ref_atom_name_chars': ref_atom_name_chars,
        }

        # get epitope region 
        collated_batch['ag_hotspot'] = get_ag_hotspot(
            collated_batch['pseudo_beta'],
            collated_batch['loop_mask'],
            collated_batch['diffuse_mask'],
            threshold=8
        ) * 0

        # Center based on motif locations
        motif_mask = 1 - collated_batch['loop_mask'] # (B, L)
        motif_1 = collated_batch['trans_1'] * motif_mask[..., None] # (B, L, 3)
        motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask, dim=1) + 1)[..., None] # (B, 3)

        collated_batch["trans_1"] = collated_batch['trans_1'] - motif_com[:, None, :] # (B, L, 3)
        collated_batch['atom14_gt_positions'] = collated_batch['atom14_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
        collated_batch['atom14_alt_gt_positions'] = collated_batch['atom14_alt_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
        collated_batch['pseudo_beta'] = collated_batch['pseudo_beta'] - motif_com[:, None, :] # (B, L, 3)

        # create atom diffuse_mask 
        atom_diffuse_mask = collated_batch['atom14_gt_exists'].clone() # (B, L, 14)
        atom_diffuse_mask[..., :3] *= collated_batch["diffuse_mask"].unsqueeze(-1)
        collated_batch['atom_diffuse_mask'] = du.atom_flatten(atom_diffuse_mask, collated_batch['atom14_gt_exists'])
        collated_batch['r_1'] = du.atom_flatten(collated_batch['atom14_gt_positions'], collated_batch['atom14_gt_exists'])
        
        # edge mask 
        collated_batch["edge_mask"] = collated_batch['res_mask'][:, None] * collated_batch['res_mask'][:, :, None]
        
        return collated_batch

    def collate_fn(self, batch):
        """
        batch: List of dicts. Each dict has keys ['batch_0', 'batch_1', 'label']
        """
        # 1. 0번 Feat 모으기
        feats_0 = [item["batch_0"] for item in batch]
        collated_0 = self._collate_single_side(feats_0)
        
        # 2. 1번 Feat 모으기
        feats_1 = [item["batch_1"] for item in batch]
        collated_1 = self._collate_single_side(feats_1)
        
        # 3. Label 모으기
        labels = [item['label'] for item in batch]
        labels = torch.tensor(labels, dtype=torch.long)
        
        kd1s = [item['kd1'] for item in batch]
        kd1s = torch.tensor(kd1s, dtype=torch.float)
        kd2s = [item['kd2'] for item in batch]
        kd2s = torch.tensor(kd2s, dtype=torch.float)
        return {
            'batch_0': collated_0,
            'batch_1': collated_1,
            'label': labels,
            'kd1': kd1s,
            'kd2': kd2s
        }
    
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

