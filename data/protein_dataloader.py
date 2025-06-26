"""Protein data loader."""
import torch
import pandas as pd
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data.motif_index import embed_relpos
from data import featurizer
from itertools import accumulate
import bisect


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

    def apply_probabilistic_mask(self, mask: torch.Tensor, masking_ratio: float) -> torch.Tensor:
        device = mask.device
        B, L = mask.shape
        prob_matrix = torch.rand((1, L), device=device)  # 배치 전체에 동일 적용
        
        new_mask = torch.where(
            (mask == 1) & (prob_matrix < masking_ratio),
            torch.ones_like(mask),
            torch.zeros_like(mask)
        )

        for i in range(B): 
            if torch.all(new_mask[i] == 0):
                new_mask[i] = mask[i]

        return new_mask

    def collate_fn(self, batch):
        cropped_batch = []
        for i, feat in enumerate(batch):
            cropped_feat = {}
            not_crop_key = ['res_idx', 'scaffold_idx', 'chain_seq_list', 'csv_idx', 'masked_chain', 'first_chain_len', 'raw_path', 'mode']

            for key in feat.keys():
                if key not in not_crop_key:
                    cropped_feat[key] = feat[key][feat['res_idx']]

                if key == 'chain_seq_list':
                    lengths = [len(s) for s in feat[key]]
                    start_positions = list(accumulate([0] + lengths))
                    merged = "".join(feat[key])
                    cropped_seq_list = [[] for _ in range(len(feat[key]))]

                    for idx in feat['res_idx']:
                        chain_idx = bisect.bisect_right(start_positions, idx) - 1
                        cropped_seq_list[chain_idx].append(merged[idx])

                    cropped_feat[key] = ["".join(chain_seq) for chain_seq in cropped_seq_list]

            relpos_emb = embed_relpos(feat['res_idx'], cropped_feat['chain_seq_list'])
            cropped_feat['pair_init'] = relpos_emb
            cropped_feat['csv_idx'] = feat['csv_idx']
            cropped_feat['res_idx'] = torch.tensor(feat['res_idx'])
            
            del cropped_feat['chain_seq_list']
            cropped_batch.append(cropped_feat)

        cropped_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}   

        for key in cropped_batch.keys():     
            cropped_batch[key] = torch.stack(cropped_batch[key], dim=0)  

        cropped_batch['mode'] = feat['mode']
        cropped_batch['raw_path'] = feat['raw_path']
        cropped_batch['original_diffuse_mask'] = cropped_batch['diffuse_mask']
        
        ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, ref_pos = \
            featurizer.get_ref_basic_feature(cropped_batch['aatype'], cropped_batch['atom14_gt_exists'], cropped_batch['res_idx'])
        cropped_batch['ref_feature_dict'] = {
            'ref_space_uid': ref_space_uid,
            'atom_to_token_idx': atom_to_token_idx,
            'ref_pos': ref_pos,
            'ref_element': ref_element,
            'ref_charge': ref_charge,
            'ref_atom_name_chars': ref_atom_name_chars
        }
        return cropped_batch

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'set_current_epoch'):
            dataset.set_current_epoch(self._current_epoch)
    
    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
            ),
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if self.loader_cfg.num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=self.loader_cfg.num_workers > 0,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if self.loader_cfg.num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )


class LengthBatcher:
    def __init__(self, *, sampler_cfg, metadata_csv, seed=123, shuffle=True):
        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log = logging.getLogger(__name__)

    def _sample_indices(self):
        if 'cluster' in self._data_csv.columns:
            random_seed = self.seed + self.epoch
            cluster_sample = self._data_csv[self._data_csv['mode'].isin(['ab', 'nanobody', 'monomer', 'polymer'])].groupby('cluster').sample(
                1, random_state=random_seed
            )
            general_df = self._data_csv[self._data_csv['mode'] == 'general']
            
            # stage 2 
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
    
        replica_csv = self._data_csv[self._data_csv['index'].isin(indices)]

        sample_order = []
        for _, row in replica_csv.iterrows():
            seq_len = row['seq_len']
            if row['mode'] in ['ab', 'nanobody'] and seq_len > self._sampler_cfg.ab_max_num_res:
                seq_len = self._sampler_cfg.ab_max_num_res
            elif row['mode'] in ['general', 'polymer', 'monomer'] and seq_len > self._sampler_cfg.general_max_num_res:
                seq_len = self._sampler_cfg.general_max_num_res

            max_batch_size = max(1, min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // (seq_len ** 2) + 1
            ))

            sample_order.append([row['index']] * max_batch_size)

        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        self.sample_order = []
        self.sample_order.extend(self._replica_epoch_batches())

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)
