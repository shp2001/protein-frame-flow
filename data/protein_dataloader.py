"""Protein data loader."""
import math
import torch
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist

from data.motif_index import embed_relpos
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
        """
        Args:
            mask: (B, L) 크기의 이진 텐서 (1=유지, 0=마스킹)
            masking_ratio: 1을 유지할 확률 (0.0 ~ 1.0)
        Returns:
            (B, L) 크기의 새 마스크 텐서
        """
        device = mask.device
        B, L = mask.shape
        # 1. [1, L] 크기의 확률 텐서 생성 → 배치 전체에 동일 적용
        prob_matrix = torch.rand((1, L), device=device)  # 차원 확장
        
        # 2. masking_ratio 기준으로 0/1 결정
        new_mask = torch.where(
            (mask == 1) & (prob_matrix < masking_ratio),  # 조건
            torch.ones_like(mask),                       # True면 1 유지
            torch.zeros_like(mask)                       # False면 0으로 마스킹
        )

        for i in range(B): 
            if torch.all(new_mask[i] == 0):
                new_mask[i] = mask[i]

        return new_mask

    def collate_fn(self, batch):
        cropped_batch = []
        # res_idxs = []
        # trans_1s = []
        # raw_paths = []
        # masking_ratio = self.trainer.datamodule.masking_ratio if hasattr(self, 'trainer') else self.masking_ratio
        for i, feat in enumerate(batch):
            # crop the feats
            # res_idxs.append(feat['res_idx'])
            # trans_1s.append(feat['diffuse_mask'])
            # raw_paths.append(feat['raw_path'])
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

            # make pair_init (relpos)
            relpos_emb = embed_relpos(feat['res_idx'],
                                    cropped_feat['chain_seq_list'])
            
            cropped_feat['pair_init'] = relpos_emb
            cropped_feat['csv_idx'] = feat['csv_idx']
            cropped_feat['res_idx'] = torch.tensor(feat['res_idx'])


            del cropped_feat['chain_seq_list']

            cropped_batch.append(cropped_feat)
            
        # all_same = all(x == res_idxs[0] for x in res_idxs)
        # all_same_trans = all(torch.equal(x, trans_1s[0]) for x in trans_1s)

        # if not all_same_trans:
        #     raise ValueError(f'{raw_paths} \n {feat["raw_path"]}: {trans_1s}')
        
        # if not all_same:
        #     raise ValueError(f'{feat["raw_path"]}: {res_idxs}')
        
        cropped_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}   

        for key in cropped_batch.keys():     
            cropped_batch[key] = torch.stack(cropped_batch[key], dim=0)  

        cropped_batch['mode'] = feat['mode']
        cropped_batch['raw_path'] = feat['raw_path']
        cropped_batch['original_diffuse_mask'] = cropped_batch['diffuse_mask']

        # masking scheduling 
        # if mask_schedule:
        #     cropped_batch['diffuse_mask'] = self.apply_probabilistic_mask(cropped_batch['diffuse_mask'], self.masking_ratio)
        return cropped_batch

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'set_current_epoch'):
            dataset.set_current_epoch(self._current_epoch)  # 이 부분 중요
    
    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=False if num_workers > 0 else False,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn
        )

    def predict_dataloader(self):
        num_workers = self.loader_cfg.num_workers
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
            collate_fn=self.collate_fn
        )


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
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
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self.ab_count = self._data_csv[self._data_csv['mode'].isin(['ab', 'nanobody'])].groupby('cluster').ngroups
        num_batches = self.ab_count * 2

        self._num_batches = num_batches
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _sample_indices(self):

        if 'cluster' in self._data_csv.keys():
            random_seed = self.seed + self.epoch
            ab_cluster_sample = self._data_csv[self._data_csv['mode'].isin(['ab', 'nanobody'])].groupby('cluster').sample(
                1, random_state=random_seed
            )
            general_sample = self._data_csv[self._data_csv['mode'] == 'general'].sample(
                self.ab_count , random_state=random_seed, replace=False
            ) 
            
            cluster_sample = pd.concat([ab_cluster_sample, general_sample])
            return cluster_sample['index'].tolist()
        
        else:
            return self._data_csv['index'].tolist()
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()

        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]
        

        replica_csv = self._data_csv.iloc[indices]
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for i in range(len(replica_csv)):
            seq_len = replica_csv.iloc[i]['seq_len']
            len_df = replica_csv.iloc[i]

            if replica_csv.iloc[i]['mode'] == 'ab' or replica_csv.iloc[i]['mode'] == 'nanobody':
                if seq_len > self._sampler_cfg.ab_max_num_res:
                    seq_len = self._sampler_cfg.ab_max_num_res # crop
            
            if replica_csv.iloc[i]['mode'] == 'general' and seq_len > self._sampler_cfg.general_max_num_res:
                seq_len = self._sampler_cfg.general_max_num_res # crop

            max_batch_size = max(1, min(  # 최소 1로 보장
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            ))

            batch_df = len_df
            batch_indices = batch_df['index']
            batch_repeats = max(1, math.floor(max_batch_size))  # 최소 1로 보장
            sample_order.append([batch_indices] * batch_repeats)

        # Remove any length bias.
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return len(self.sample_order)
