"""Protein data loader."""
import torch
import pandas as pd
import logging

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist

from data.motif_index import embed_relpos
from data import featurizer
from data import utils as du

from itertools import accumulate
import bisect
import math 

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
        if hasattr(self._train_dataset, 'set_current_epoch'):
            self._train_dataset.set_current_epoch(epoch)


    def collate_fn(self, batch):
        cropped_batch = []
        for i, feat in enumerate(batch):
            cropped_feat = {}
            not_crop_key = ['crop_idx', 'scaffold_idx', 'chain_seq_list', 'csv_idx', 'masked_chain', 'first_chain_len', 'raw_path', 'mode']
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

            relpos_emb, asym_id, entity_id, sym_id = embed_relpos(cropped_feat['residue_index'], cropped_feat['chain_seq_list'])
            cropped_feat['pair_init'] = relpos_emb
            cropped_feat['csv_idx'] = feat['csv_idx']
            cropped_feat['crop_idx'] = torch.tensor(feat['crop_idx'])
            
            del cropped_feat['chain_seq_list']
            cropped_batch.append(cropped_feat)

        cropped_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}   

        for key in cropped_batch.keys():
            cropped_batch[key] = torch.stack(cropped_batch[key], dim=0)  

        cropped_batch['mode'] = feat['mode']
        cropped_batch['raw_path'] = feat['raw_path']
        if cropped_batch['diffuse_mask'][0, 0] == 1:
            cropped_batch['diffuse_mask'][:, 0] = 0
        if cropped_batch['diffuse_mask'][0, -1] == 1:
            cropped_batch['diffuse_mask'][:, -1] = 0
        
        ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, ref_pos, ref_rigid_frame = \
            featurizer.get_ref_basic_feature(cropped_batch['aatype'], cropped_batch['atom14_gt_exists'], cropped_batch['residue_index'])
        cropped_batch['ref_feature_dict'] = {
            'ref_space_uid': ref_space_uid,
            'atom_to_token_idx': atom_to_token_idx,
            'ref_pos': ref_pos,
            'ref_element': ref_element,
            'ref_charge': ref_charge,
            'ref_atom_name_chars': ref_atom_name_chars,
            'ref_rigid_frame': ref_rigid_frame
        }

        # Center based on motif locations
        motif_mask = 1 - cropped_batch['diffuse_mask'] # (B, L)
        motif_1 = cropped_batch['trans_1'] * motif_mask[..., None] # (B, L, 3)
        motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask, dim=1) + 1)[..., None] # (B, 3)

        cropped_batch["trans_1"] = cropped_batch['trans_1'] - motif_com[:, None, :] # (B, L, 3)
        cropped_batch['atom14_gt_positions'] = cropped_batch['atom14_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
        cropped_batch['atom14_alt_gt_positions'] = cropped_batch['atom14_alt_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
        cropped_batch['pseudo_beta'] = cropped_batch['pseudo_beta'] - motif_com[:, None, :] # (B, L, 3)
        
        cropped_batch['original_diffuse_mask'] = torch.tensor(feat['diffuse_mask']).to(motif_mask.device)
        cropped_batch['original_loop_mask'] = torch.tensor(feat['loop_mask']).to(motif_mask.device)
        return cropped_batch

    
    def train_dataloader(self, rank=None, num_replicas=None):
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas
            ),
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if self.loader_cfg.num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if self.loader_cfg.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
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


class LengthBatcher:
    def __init__(self, 
                 *, 
                 sampler_cfg, 
                 metadata_csv, 
                 seed=123, 
                 shuffle=True,
                 num_replicas=None,
                 rank=None):
        super().__init__()
        self._log = logging.getLogger(__name__)

        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        
        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        

    def _sample_indices(self):
        if 'cluster' in self._data_csv.columns:
            random_seed = self.seed + self.epoch
            cluster_sample = self._data_csv[self._data_csv['mode'].isin(['ab', 'nanobody', 'polymer'])].groupby('cluster').sample(
                1, random_state=random_seed
            )
            # stage 1 
            monomer_df = self._data_csv[self._data_csv['mode'] == 'monomer'] 
            monomer_sample = monomer_df.groupby('cluster').sample(
                1, random_state=random_seed
            )
            if len(monomer_df) > cluster_sample.shape[0]:
                monomer_sample = monomer_sample.sample(
                    len(cluster_sample), random_state=random_seed, replace=False
                )
                print(f"sampled_monomer", len(monomer_sample['cluster']))
                cluster_sample = pd.concat([cluster_sample, monomer_sample])
                 
            # stage 2
            general_df = self._data_csv[self._data_csv['mode'] == 'general'] 
            if len(general_df) > cluster_sample.shape[0]: 
                general_sample = self._data_csv[self._data_csv['mode'] == 'general'].sample(
                    cluster_sample.shape[0], random_state=random_seed, replace=False
                )
                cluster_sample = pd.concat([cluster_sample, general_sample])
            
            index_list = cluster_sample['index'].tolist()
            self._num_batches = len(index_list) 
            return index_list


    def _replica_epoch_batches(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).tolist()
            indices = [indices[i] for i in new_order]

        # rank 별로 분할 
        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[indices[self.rank::self.num_replicas]]
        else:
            replica_csv = self._data_csv

        sample_order = []

        # 길이별로 max_batch_size 설정
        for _, row in replica_csv.iterrows():
            seq_len = row['seq_len']
            if row['mode'] in ['ab', 'nanobody'] and seq_len > self._sampler_cfg.ab_max_num_res:
                seq_len = self._sampler_cfg.ab_max_num_res
            elif row['mode'] in ['general', 'polymer', 'monomer'] and seq_len > self._sampler_cfg.general_max_num_res:
                seq_len = self._sampler_cfg.general_max_num_res
 
            max_batch_size = max(1, min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // (seq_len ** 2) + 1
            )) # at least 4 

            sample_order.append([row['index']] * max_batch_size)

        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        self.sample_order = []
        self.sample_order.extend(self._replica_epoch_batches())

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)
