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

    def _collate_single_side(self, batch_feats):
        """
        feats_paired의 0번 혹은 1번 키에 해당하는 feature list를 처리하는 함수.
        기존 collate_fn 로직을 이곳으로 옮김.
        """
        cropped_batch = []
        for feat in batch_feats:
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

            asym_id, entity_id, sym_id = get_relpos_input(cropped_feat['chain_seq_list'])
            cropped_feat['asym_id'] = torch.tensor(asym_id)
            cropped_feat['entity_id'] = torch.tensor(entity_id)
            cropped_feat['sym_id'] = torch.tensor(sym_id)
            cropped_feat['crop_idx'] = torch.tensor(feat['crop_idx'])
            cropped_feat['csv_idx'] = feat['csv_idx']
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
        collated_batch['raw_path'] = [f['raw_path'] for f in batch_feats]

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
        )

        # Center based on motif locations
        motif_mask = 1 - collated_batch['loop_mask'] # (B, L)
        motif_1 = collated_batch['trans_1'] * motif_mask[..., None] # (B, L, 3)
        motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask, dim=1) + 1)[..., None] # (B, 3)

        collated_batch["trans_1"] = collated_batch['trans_1'] - motif_com[:, None, :] # (B, L, 3)
        collated_batch['atom14_gt_positions'] = collated_batch['atom14_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
        collated_batch['atom14_alt_gt_positions'] = collated_batch['atom14_alt_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
        collated_batch['pseudo_beta'] = collated_batch['pseudo_beta'] - motif_com[:, None, :] # (B, L, 3)
        
        # get interface index 
        if collated_batch['mode'] not in ["monomer", "polymer"]:
            # 배치 처리를 위해 loop를 돌거나 au.get_cdr_and_neighbors가 배치 처리를 지원해야 함.
            # 기존 코드는 cropped_batch['loop_mask'][0]를 써서 첫 샘플 기준이었음.
            # 정확성을 위해 여기서는 첫 번째 샘플 기준으로 계산하거나, 필요 시 배치 전체 루프 구현 필요.
            # 여기서는 기존 로직 유지 (첫번째 샘플 기준)
            cdr_residues, neighbor_indices, anchor_residues = au.get_cdr_and_neighbors(
                collated_batch['atom14_gt_positions'],
                collated_batch['atom14_gt_exists'],
                collated_batch['loop_mask'][0],
                collated_batch['mode']
            )
            collated_batch['cdr_residues'] = cdr_residues
            collated_batch['neighbor_indices'] = neighbor_indices
            collated_batch['anchor_residues'] = anchor_residues

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
        batch: List of dicts. Each dict has keys [0, 1, 'label']
        """
        # 1. 0번 Feat 모으기
        feats_0 = [item[0] for item in batch]
        collated_0 = self._collate_single_side(feats_0)
        
        # 2. 1번 Feat 모으기
        feats_1 = [item[1] for item in batch]
        collated_1 = self._collate_single_side(feats_1)
        
        # 3. Label 모으기
        labels = [item['label'] for item in batch]
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            0: collated_0,
            1: collated_1,
            'label': labels
        }
    
    def train_dataloader(self, rank=None, num_replicas=None):
        return DataLoader(
            self._train_dataset,
            batch_sampler=LengthBatcher(
                sampler_cfg=self.sampler_cfg,
                metadata_csv=self._train_dataset.csv,
                rank=rank,
                num_replicas=num_replicas,
            ),
            num_workers=self.loader_cfg.num_workers,
            prefetch_factor=None if self.loader_cfg.num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if self.loader_cfg.num_workers > 0 else False,
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
            if not monomer_df.empty:
                monomer_sample = monomer_df.groupby('cluster').sample(
                    1, random_state=random_seed
                )
                if len(monomer_df) > cluster_sample.shape[0]:
                    monomer_sample = monomer_sample.sample(
                        len(cluster_sample), random_state=random_seed, replace=False
                    )
                    # print(f"sampled_monomer", len(monomer_sample['cluster']))
                    cluster_sample = pd.concat([cluster_sample, monomer_sample])
                 
            # stage 2
            general_df = self._data_csv[self._data_csv['mode'] == 'general'] 
            if len(general_df) > cluster_sample.shape[0]: 
                general_sample = self._data_csv[self._data_csv['mode'] == 'general'].sample(
                    cluster_sample.shape[0], random_state=random_seed, replace=False
                )
                cluster_sample = pd.concat([cluster_sample, general_sample])
            
            index_list = cluster_sample['index'].tolist()
            return index_list
        else:
            # cluster 정보가 없다면 전체 인덱스 반환
            return self._data_csv['index'].tolist()


    def _replica_epoch_batches(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        
        # 1. 사용할 인덱스 추출
        indices = self._sample_indices()

        # 2. 인덱스 셔플 (랜덤성 부여) 후 Rank 분배
        #    참고: Bucketing을 하려면 길이 순 정렬이 필요하지만, 
        #    전체 데이터셋에서 랜덤하게 Rank에 할당된 부분집합을 가져온 뒤 그 안에서 정렬하는 것이 일반적임.
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).tolist()
            indices = [indices[i] for i in new_order]

        # Rank에 맞는 부분 데이터만 가져오기
        if len(indices) > self.num_replicas:
            replica_indices = indices[self.rank::self.num_replicas]
        else:
            replica_indices = indices

        # 3. 데이터프레임에서 해당 인덱스들 가져와서 '길이(seq_len)' 순으로 정렬 (내림차순)
        #    내림차순 정렬 시 가장 긴 시퀀스 기준으로 배치가 형성되므로 padding issue나 OOM 방지에 유리
        subset = self._data_csv.loc[replica_indices].sort_values('seq_len', ascending=False)

        batches = []
        current_batch = []
        current_max_bs = -1

        # 4. 정렬된 데이터를 순회하며 Dynamic Batching 수행
        for _, row in subset.iterrows():
            seq_len = row['seq_len']
            
            # 길이 제한 적용 (기존 로직 유지)
            if row['mode'] in ['ab', 'nanobody'] and seq_len > self._sampler_cfg.ab_max_num_res:
                seq_len = self._sampler_cfg.ab_max_num_res
            elif row['mode'] in ['general', 'polymer', 'monomer'] and seq_len > self._sampler_cfg.general_max_num_res:
                seq_len = self._sampler_cfg.general_max_num_res

            # 현재 샘플의 길이를 기준으로 최대 배치 크기 계산
            # 길이가 길수록 max_batch_size는 작아짐
            sample_max_bs = max(1, min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // (seq_len ** 2) + 1
            ))

            # 첫 샘플이거나, 새로운 배치가 시작될 때 기준 Batch Size 설정
            if not current_batch:
                current_max_bs = sample_max_bs
            
            # 현재 배치의 기준 크기는 배치 내 가장 긴 샘플(정렬했으므로 첫번째)에 의해 결정되거나,
            # 안전을 위해 현재 샘플의 max_bs와 비교하여 더 보수적인 값을 취할 수도 있음.
            # 여기서는 내림차순 정렬되어 있으므로, current_max_bs(배치 시작 시점의 max_bs)가 
            # 뒤에 오는 샘플들(더 짧음 -> 더 큰 허용 배수)보다 작거나 같으므로 safe함.
            
            current_batch.append(row['index'])

            # 배치 사이즈가 꽉 차면 결과에 추가하고 초기화
            if len(current_batch) >= current_max_bs:
                batches.append(current_batch)
                current_batch = []
                current_max_bs = -1

        # 남은 자투리 데이터 처리
        if current_batch:
            batches.append(current_batch)

        # 5. 생성된 배치들의 순서를 셔플
        if self.shuffle:
            # 배치 내부(샘플끼리)는 길이순 정렬 유지, 배치 간의 순서만 섞음
            new_order = torch.randperm(len(batches), generator=rng).numpy().tolist()
            return [batches[i] for i in new_order]
        
        return batches

    def _create_batches(self):
        self.sample_order = []
        self.sample_order.extend(self._replica_epoch_batches())

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        # __len__은 DataLoader 초기화 시에는 정확히 알 수 없고 epoch마다 달라질 수 있음.
        # 가장 최근 생성된 sample_order 길이를 반환하거나, 근사치를 반환해야 함.
        # PyTorch Lightning은 이 값을 기준으로 진행률 표시줄(tqdm)을 그림.
        if not hasattr(self, 'sample_order'):
             self._create_batches()
        return len(self.sample_order)