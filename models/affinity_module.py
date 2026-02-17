from typing import Any
import torch
import time
import math 

import os
import random
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from analysis import utils as au
from models.flow_model import FlowModel
from models.affinity_model import AffinityHead

from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from experiments import utils as eu
from models.loss import *
import sys 
import json 

sys.stdout.flush()

class AffinityModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)
        self._affinity_cfg = cfg.model.affinity_head
        self.affinity_model = AffinityHead(
            n_blocks=self._affinity_cfg.n_blocks,
            c_z=self._affinity_cfg.c_z,
            c_s_inputs=self._affinity_cfg.c_s_inputs,
            pairformer_dropout=self._affinity_cfg.pairformer_dropout,
            blocks_per_ckpt=self._affinity_cfg.blocks_per_ckpt,
            distance_bin_start=self._affinity_cfg.distance_bin_start,
            distance_bin_end=self._affinity_cfg.distance_bin_end,
            distance_bin_step=self._affinity_cfg.distance_bin_step,
            stop_gradient=self._affinity_cfg.stop_gradient,
        )
        # Set-up interpolant for mini-rollout
        self.rollout = Interpolant(cfg.rollout)
        # Set-up interpolant for validation or prediction
        self.interpolant = Interpolant(cfg.interpolant)

        self.learning_rate = self._exp_cfg.optimizer.max_lr
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None
        self.save_file = True
        self.is_loss_nan = False
        
    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    # def on_train_batch_start(self, batch, batch_idx):
    #     optimizer = self.trainer.optimizers[0]
    #     current_lr = optimizer.param_groups[0]['lr']
    #     self.log(
    #         'lr',
    #         current_lr,
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=False
    #     )
    # def on_train_batch_start(self, batch, batch_idx):
    #     # 모든 학습 가능한 파라미터 초기화
    #     for n, p in self.named_parameters():
    #         print(f"{n}: {p.requires_grad}")
    #     for p in self.parameters():
    #         if p.requires_grad:
    #             p.grad = None

    #     # Forward pass 전 파라미터 기록 (메모리 주소까지 추적)
    #     self._params_before = {id(p): n for n, p in self.named_parameters() if p.requires_grad}

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     # Backward 이후 gradient가 계산된 파라미터 추적
    #     grads = {}
    #     for n, p in self.named_parameters():
    #         if p.requires_grad and p.grad is not None:
    #             grads[id(p)] = n
        
    #     # 사용되지 않은 파라미터 찾기
    #     unused = [self._params_before[id_p] for id_p in self._params_before 
    #             if id_p not in grads]
    #     used = [self._params_before[id_p] for id_p in self._params_before 
    #             if id_p in grads]
    #     if used:
    #         print(f"✅ 실제로 사용된 파라미터: {used}")
    #     if unused:
    #         print(f"🔥 실제로 사용되지 않은 파라미터: {unused}")

    #     else:
    #         print("✅ 모든 파라미터가 사용되었습니다.")

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, paired_batch, N_cycle):
        affinity_pred_values = []
        affinity_pred_logits = []

        # 1. Unbound 모드 판별 (batch_0 내부 구조 확인)
        use_unbound = self._data_cfg.use_unbound
        for i in range(2):
            full_sample = paired_batch[f"batch_{i}"]
            batch_complex = full_sample['complex']
            ligand_mask = batch_complex['ligand_mask'].bool()
            inter_mask = (ligand_mask.unsqueeze(2) != ligand_mask.unsqueeze(1)).float()

            is_ligand = ligand_mask
            is_receptor = ~is_ligand

            # rollout for affinity
            self.rollout.set_device(batch_complex['edge_mask'].device)

            with torch.no_grad():
                if use_unbound:
                    batch_ligand = full_sample['ligand']
                    batch_receptor = full_sample['receptor']
                    
                    s_init_complex, z_init_complex, _ = self.model.embed_input(batch_complex)
                    s_init_ligand, z_init_ligand, _ = self.model.embed_input(batch_ligand)
                    s_init_receptor, z_init_receptor, _ = self.model.embed_input(batch_receptor)

                    # 2. Pairformer Pass (Complex)
                    _, s_complex, z_complex, _ = self.model.do_pairformer(
                        s_init_complex, z_init_complex, 
                        batch_complex['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_complex['edge_mask'].shape[0]
                    )
                    # 3. Pairformer Pass (Ligand)
                    _, _, z_ligand, _ = self.model.do_pairformer(
                        s_init_ligand, z_init_ligand, 
                        batch_ligand['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_ligand['edge_mask'].shape[0]
                    )
                    # 4. Pairformer Pass (Receptor)
                    _, _, z_receptor, _ = self.model.do_pairformer(
                        s_init_receptor, z_init_receptor, 
                        batch_receptor['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_receptor['edge_mask'].shape[0]
                    )

                    # --- Unbound Feature Mapping & Concatenation ---
                    B, L, _, C = z_complex.shape
                    
                    # 1. Unbound 정보를 담을 텐서 생성 (모든 값을 0으로 초기화)
                    # [B, L_complex, L_complex, 128]
                    z_unbound = torch.zeros_like(z_complex)

                    L_lig_tensor = z_ligand.shape[1]
                    L_rec_tensor = z_receptor.shape[1]

                    for b in range(B):
                        lig_idx = is_ligand[b].nonzero(as_tuple=True)[0]
                        rec_idx = is_receptor[b].nonzero(as_tuple=True)[0]

                        n_lig = len(lig_idx)
                        n_rec = len(rec_idx)

                        # --- Ligand 영역 채우기 ---
                        if n_lig != L_lig_tensor:
                            raise RuntimeError(f"Ligand dimension mismatch: mask has {n_lig}, tensor has {L_lig_tensor}")
                        # Ligand 영역 (Intra)에 z_ligand 값 할당
                        z_unbound[b][lig_idx[:, None], lig_idx[None, :]] = z_ligand[b]
                        # --- Receptor 영역 채우기 ---
                        if n_rec != L_rec_tensor:
                            raise RuntimeError(f"Receptor dimension mismatch: mask has {n_rec}, tensor has {L_rec_tensor}")
                        # Receptor 영역 (Intra)에 z_receptor 값 할당
                        z_unbound[b][rec_idx[:, None], rec_idx[None, :]] = z_receptor[b]

                else:
                    s_init_complex, z_init_complex, _ = self.model.embed_input(batch_complex)
                    _, s, z, pair_outputs = self.model.do_pairformer(
                        s_init_complex, 
                        z_init_complex, 
                        batch_complex['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_complex['edge_mask'].shape[0]
                        )
                    z_complex = z
                    s_complex = s
                    
                pred_trans_1 = None
                if self._affinity_cfg.use_coords:
                    _, _, pred_positions, pred_trans_1 = self.rollout.sample(
                        self.model,
                        batch_complex,
                        s_complex,
                        s_complex,
                        z_complex
                    )
            
            affinity_pred_value, affinity_pred_logit = self.affinity_model(
                s_inputs=s_init_complex[0],
                s_trunk=s_complex[0],
                z_bound=z_complex[0],
                z_unbound=z_unbound[0],
                inter_mask=inter_mask[0],
                loop_mask=batch_complex['loop_mask'][0],
                edge_mask=batch_complex['edge_mask'][0],
                x_pred_coords=pred_trans_1,
                )       
            affinity_pred_values.append(affinity_pred_value)
            affinity_pred_logits.append(affinity_pred_logit)
        
        # 텐서 이동
        device = batch_complex['edge_mask'].device
        label = paired_batch['label'].to(device)
        kd1 = paired_batch["kd1"].to(device)
        kd2 = paired_batch["kd2"].to(device)
        
        # calculate affinity rank loss 
        target = 1.0 - 2.0 * label.float()
        margin_loss_fn = nn.MarginRankingLoss(margin=self._exp_cfg.training.rank_margin, reduction='none')
        affinity_rank_loss = margin_loss_fn(affinity_pred_values[0], affinity_pred_values[1], target)
        if len(affinity_rank_loss.shape) == 1:
            affinity_rank_loss = affinity_rank_loss[None, ...]
        affinity_rank_loss = torch.clamp(affinity_rank_loss, max=3.0)
        
        # calculate affinity regression loss 
        preds = torch.stack(affinity_pred_values) # (2, B)
        targets = torch.stack([torch.log10(kd1), torch.log10(kd2)]) # (2, B)
        mask = ~torch.isinf(targets)
        
        if mask.any():
            diff = torch.abs(preds[mask] - targets[mask])
            affinity_reg_loss = torch.mean(nn.functional.relu(diff - self._exp_cfg.training.reg_margin))
        else:
            affinity_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        if len(affinity_reg_loss.shape) == 0: # scalar check
            affinity_reg_loss = affinity_reg_loss.unsqueeze(0)

        # calculate affinity probability loss (Non-binder detection)
        # inf가 하나라도 포함된 샘플에 대해 cross entropy 계산
        is_inf_mask = torch.isinf(kd1) | torch.isinf(kd2)
        if is_inf_mask.any():
            target_0 = label.float()
            target_1 = 1.0 - label.float()
            loss_0 = nn.functional.binary_cross_entropy_with_logits(affinity_pred_logits[0].float(), target_0, reduction='none')
            loss_1 = nn.functional.binary_cross_entropy_with_logits(affinity_pred_logits[1].float(), target_1, reduction='none')
            
            # inf인 샘플만 loss 반영, 나머지는 0
            affinity_binding_prob_loss = (loss_0 + loss_1) / 2
            affinity_binding_prob_loss = affinity_binding_prob_loss * is_inf_mask.float()
        else:
            affinity_binding_prob_loss = torch.zeros_like(label, dtype=torch.float32)

        if len(affinity_binding_prob_loss.shape) == 1:
            affinity_binding_prob_loss = affinity_binding_prob_loss[None, ...]

        total_loss = (affinity_rank_loss.mean() * self._exp_cfg.training.rank_loss_weight + 
                      affinity_reg_loss.mean() * self._exp_cfg.training.reg_loss_weight + 
                      affinity_binding_prob_loss.mean() * self._exp_cfg.training.prob_loss_weight)

        print("--------------------------------")
        print("affinity_pred_values[0]", affinity_pred_values[0])
        print("affinity_pred_values[1]", affinity_pred_values[1])
        print("log_kd1", torch.log10(paired_batch['kd1']))
        print("log_kd2", torch.log10(paired_batch['kd2']))
        print("--------------------------------")

        return {
            "total_loss": total_loss,
            "affinity_rank_loss": affinity_rank_loss,
            "affinity_reg_loss": affinity_reg_loss,
            "affinity_binding_prob_loss": affinity_binding_prob_loss,
        }

    def validation_step(self, paired_batch: Any, batch_idx: int):
        batch_metrics = []
        affinity_pred_values = []
        affinity_pred_logits = []

        for i in range(2):
            # rollout for affinity
            use_unbound = self._data_cfg.use_unbound
            full_sample = paired_batch[f"batch_{i}"]
            batch_complex = full_sample['complex']
            
            loop_mask_complex = batch_complex['loop_mask'] # (B, L)
            ligand_mask = batch_complex['ligand_mask']
            inter_mask = (ligand_mask.unsqueeze(2) != ligand_mask.unsqueeze(1)).float()

            self.rollout.set_device(loop_mask_complex.device)
            num_batch, num_res = loop_mask_complex.shape

            with torch.no_grad():
                if use_unbound:
                    batch_ligand = full_sample['ligand']
                    batch_receptor = full_sample['receptor']

                    s_init_complex, z_init_complex, _ = self.model.embed_input(batch_complex)
                    s_init_ligand, z_init_ligand, _ = self.model.embed_input(batch_ligand)
                    s_init_receptor, z_init_receptor, _ = self.model.embed_input(batch_receptor)

                    # 2. Pairformer Pass (Complex)
                    _, s_complex, z_complex, _ = self.model.do_pairformer(
                        s_init_complex, z_init_complex, 
                        batch_complex['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_complex['edge_mask'].shape[0]
                    )
                    # 3. Pairformer Pass (Ligand)
                    _, _, z_ligand, _ = self.model.do_pairformer(
                        s_init_ligand, z_init_ligand, 
                        batch_ligand['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_ligand['edge_mask'].shape[0]
                    )
                    # 4. Pairformer Pass (Receptor)
                    _, _, z_receptor, _ = self.model.do_pairformer(
                        s_init_receptor, z_init_receptor, 
                        batch_receptor['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_receptor['edge_mask'].shape[0]
                    )
                    # --- Unbound Feature Mapping & Concatenation ---
                    B, L, _, C = z_complex.shape
                    
                    # 1. Unbound 정보를 담을 텐서 생성 (모든 값을 0으로 초기화)
                    # [B, L_complex, L_complex, 128]
                    z_unbound = torch.zeros_like(z_complex)

                    ligand_mask = batch_complex['ligand_mask'].bool()
                    is_ligand = ligand_mask
                    is_receptor = ~is_ligand

                    L_lig_tensor = z_ligand.shape[1]
                    L_rec_tensor = z_receptor.shape[1]

                    for b in range(B):
                        lig_idx = is_ligand[b].nonzero(as_tuple=True)[0]
                        rec_idx = is_receptor[b].nonzero(as_tuple=True)[0]

                        n_lig = len(lig_idx)
                        n_rec = len(rec_idx)

                        # --- Ligand 영역 채우기 ---
                        if n_lig != L_lig_tensor:
                            raise RuntimeError(f"Ligand dimension mismatch: mask has {n_lig}, tensor has {L_lig_tensor}")
                        # Ligand 영역 (Intra)에 z_ligand 값 할당
                        z_unbound[b][lig_idx[:, None], lig_idx[None, :]] = z_ligand[b]
                        # --- Receptor 영역 채우기 ---
                        if n_rec != L_rec_tensor:
                            raise RuntimeError(f"Receptor dimension mismatch: mask has {n_rec}, tensor has {L_rec_tensor}")
                        # Receptor 영역 (Intra)에 z_receptor 값 할당
                        z_unbound[b][rec_idx[:, None], rec_idx[None, :]] = z_receptor[b]

                else:
                    # --- Bound 모드: 기존 단일 Complex 처리 ---
                    s_init_complex, z_init_complex, _ = self.model.embed_input(batch_complex)
                    _, s, z, _ = self.model.do_pairformer(
                        s_init_complex, z_init_complex, 
                        batch_complex['edge_mask'][0][None, ...], 
                        self._model_cfg.pairformer.n_cycles, 
                        batch_complex['edge_mask'].shape[0]
                    )
                    z_complex = z
                    s_complex = s
                # --- 공통: Coordinate Sampling (Rollout) ---
                _, _, pred_positions, pred_trans_1 = self.rollout.sample(
                    self.model,
                    batch_complex,
                    s_complex,
                    s_complex,
                    z_complex
                ) # save 3d structure for validation 
                pred_positions_37 = []
                pred_positions = du.to_numpy(pred_positions)
                for i in range(pred_positions.shape[0]):
                    pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch_complex)
                    pred_positions_37.append(pred_position_37)
                
                pred_positions = np.stack(pred_positions_37)
                
                pdb_id = os.path.basename(batch_complex['raw_path']).split('_')[0]
                sample_dir = os.path.join(
                    self.checkpoint_dir,
                    f'{pdb_id}_len_{num_res}'
                )
                os.makedirs(sample_dir, exist_ok=True)

                b_factor_alt = loop_mask_complex.cpu().numpy()
                b_factors = np.tile((b_factor_alt * 100)[:, :, None], (1, 1, 37)) # (B, L, 37)
                    
                for j in range(num_batch):
                    # Write out sample to PDB file (wo b-factors)
                    final_pos = pred_positions[j]
                    b_factor = b_factors[j]

                    saved_path = au.write_prot_to_pdb(
                        final_pos,
                        file_path=os.path.join(sample_dir, pdb_id+'.pdb'),
                        aatype=batch_complex['aatype'].cpu(),
                        chain_index=batch_complex['chain_idx'].cpu(),
                        no_indexing=False,
                        overwrite=True,
                        b_factors=b_factor
                    )

                # affinity prediction 
                affinity_pred_value, affinity_pred_logit = self.affinity_model(
                    s_inputs=s_init_complex[0],
                    s_trunk=s_complex[0],
                    z_bound=z_complex[0],
                    z_unbound=z_unbound[0],
                    inter_mask=inter_mask[0],
                    loop_mask=batch_complex['loop_mask'][0],
                    edge_mask=batch_complex['edge_mask'][0],
                    x_pred_coords=pred_trans_1,
                    ) 
    
                affinity_pred_values.append(affinity_pred_value)
                affinity_pred_logits.append(affinity_pred_logit)


        # calculate affinity loss 
        label = paired_batch['label'].to(batch_complex['edge_mask'].device)
        kd1 = paired_batch["kd1"].to(batch_complex['edge_mask'].device)
        kd2 = paired_batch["kd2"].to(batch_complex['edge_mask'].device)

        # calculate affinity rank loss 
        target = 1.0 - 2.0 * label
        margin_loss_fn = nn.MarginRankingLoss(margin=self._exp_cfg.training.rank_margin, reduction='none')
        affinity_rank_loss = margin_loss_fn(affinity_pred_values[0], affinity_pred_values[1], target)      # batch_0 > batch_1 * 10 -> label: 0
        if len(affinity_rank_loss.shape) == 1:
            affinity_rank_loss = affinity_rank_loss[None, ...]
        
        # calculate affinity regression loss 
        preds = torch.stack(affinity_pred_values)
        targets = torch.stack([torch.log10(kd1), torch.log10(kd2)])
        inf_mask = ~torch.isinf(targets)
        diff = torch.abs(preds[inf_mask] - targets[inf_mask])
        affinity_reg_loss = torch.mean(nn.functional.relu(diff - self._exp_cfg.training.reg_margin))
        if len(affinity_reg_loss.shape) == 1:
            affinity_reg_loss = affinity_reg_loss[None, ...]

        # calculate affinity probability loss 
        is_inf_mask = torch.isinf(kd1) | torch.isinf(kd2)
        if is_inf_mask.any():
            target_0 = label.float()
            target_1 = 1.0 - label 
            loss_0 = nn.functional.binary_cross_entropy_with_logits(affinity_pred_logits[0].float(), target_0, reduction='none')
            loss_1 = nn.functional.binary_cross_entropy_with_logits(affinity_pred_logits[1].float(), target_1, reduction='none')
            affinity_binding_prob_loss = (loss_0 + loss_1) / 2
            if len(affinity_binding_prob_loss.shape) == 1:
                affinity_binding_prob_loss = affinity_binding_prob_loss[None, ...]
        else:
            affinity_binding_prob_loss = torch.zeros_like(label, dtype=torch.float32, requires_grad=True)

        affinity_loss_dict = {'affinity_rank_loss': affinity_rank_loss,
                              'affinity_reg_loss': affinity_reg_loss,
                              'affinity_binding_prob_loss': affinity_binding_prob_loss}
        batch_metrics.append(affinity_loss_dict)

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)


    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)

        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
                sync_dist=True,
                rank_zero_only=False
            )
        self.validation_epoch_metrics.clear()
        torch.cuda.empty_cache()
    # def on_after_backward(self):
    #     # 모든 파라미터에 대해 gradient 값 확인
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             # 클리핑 전 gradient의 norm 출력
    #             grad_norm_before = param.grad.norm(2).item()
    #             print(f"Gradient norm for {name} before clipping: {grad_norm_before}")
                
    #             # 클리핑 후 gradient의 norm 출력
    #             torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    #             grad_norm_after = param.grad.norm(2).item()
    #             print(f"Gradient norm for {name} after clipping: {grad_norm_after}")

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )
    
    def training_step(self, paired_batch: Any, stage: int):
        self.interpolant.set_device(paired_batch['label'].device)
        N_cycle = random.randint(1, self._model_cfg.pairformer.n_cycles)
        batch_losses = self.model_step(paired_batch, N_cycle)

        # for error proof
        self.is_loss_nan = False 
        if torch.any(torch.isnan(batch_losses['total_loss'])):
            self.is_loss_nan = True
            return None 
        
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False)

        train_loss = total_losses['total_loss']

        return train_loss

    def get_cosine_scheduler_w_warmup(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 5e3,
        total_steps: int = 5e6,
        min_lr: float = 1e-5,
        max_lr: float = 1e-4,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Get cosine annealing scheduler with warmup."""

        def steplr_with_warmup(step):
            if step < warmup_steps:
                return step / warmup_steps
            elif step <= total_steps:
                theta = (step - warmup_steps) / (total_steps - warmup_steps) * math.pi
                cosine_decay = 0.5 * (1 + math.cos(theta))
                return (min_lr / max_lr) + cosine_decay * (1 - (min_lr / max_lr))
            else:
                return min_lr / max_lr  # max_lr 기준의 비율

        return torch.optim.lr_scheduler.LambdaLR(optimizer, steplr_with_warmup)
    

    def configure_optimizers(self):
        parameters = self.affinity_model.parameters()
        optimizer = torch.optim.AdamW(
            parameters, self.learning_rate, weight_decay=0.01
        )
        scheduler = self.get_cosine_scheduler_w_warmup(
            optimizer,
            self._exp_cfg.optimizer.warmup_steps,
            self._exp_cfg.optimizer.decay_steps,
            self._exp_cfg.optimizer.min_lr,
            self._exp_cfg.optimizer.max_lr,
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                }
            }

    def on_predict_start(self):
        self.pairformer_cache = {}
        self.current_pdb_id = None
        print("Pairformer cache has been initialized.")

    def predict_step(self, batch, batch_idx):
        use_unbound = self._infer_cfg.use_unbound
        batch_complex = batch["complex"]
        loop_mask_complex = batch_complex['loop_mask'] # (B, L)
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(loop_mask_complex.device)

        with torch.no_grad():
            if use_unbound:
                batch_ligand = batch['ligand']
                batch_receptor = batch['receptor']
                
                s_init_complex, z_init_complex, _ = self.model.embed_input(batch_complex)
                s_init_ligand, z_init_ligand, _ = self.model.embed_input(batch_ligand)
                s_init_receptor, z_init_receptor, _ = self.model.embed_input(batch_receptor)

                # 2. Pairformer Pass (Complex)
                _, s_complex, z_complex, _ = self.model.do_pairformer(
                    s_init_complex, z_init_complex, 
                    batch_complex['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    batch_complex['edge_mask'].shape[0]
                )
                # 3. Pairformer Pass (Ligand)
                _, _, z_ligand, _ = self.model.do_pairformer(
                    s_init_ligand, z_init_ligand, 
                    batch_ligand['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    batch_ligand['edge_mask'].shape[0]
                )
                # 4. Pairformer Pass (Receptor)
                _, _, z_receptor, _ = self.model.do_pairformer(
                    s_init_receptor, z_init_receptor, 
                    batch_receptor['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    batch_receptor['edge_mask'].shape[0]
                )
                # --- Unbound Feature Mapping & Concatenation ---
                B, L, _, C = z_complex.shape
                
                # 1. Unbound 정보를 담을 텐서 생성 (모든 값을 0으로 초기화)
                # [B, L_complex, L_complex, 128]
                z_unbound = torch.zeros_like(z_complex)

                ligand_mask = batch_complex['ligand_mask'].bool()
                inter_mask = (ligand_mask.unsqueeze(2) != ligand_mask.unsqueeze(1)).float()

                is_ligand = ligand_mask
                is_receptor = ~is_ligand

                L_lig_tensor = z_ligand.shape[1]
                L_rec_tensor = z_receptor.shape[1]

                for b in range(B):
                    lig_idx = is_ligand[b].nonzero(as_tuple=True)[0]
                    rec_idx = is_receptor[b].nonzero(as_tuple=True)[0]

                    n_lig = len(lig_idx)
                    n_rec = len(rec_idx)

                    # --- Ligand 영역 채우기 ---
                    if n_lig != L_lig_tensor:
                        raise RuntimeError(f"Ligand dimension mismatch: mask has {n_lig}, tensor has {L_lig_tensor}")
                    # Ligand 영역 (Intra)에 z_ligand 값 할당
                    z_unbound[b][lig_idx[:, None], lig_idx[None, :]] = z_ligand[b]
                    # --- Receptor 영역 채우기 ---
                    if n_rec != L_rec_tensor:
                        raise RuntimeError(f"Receptor dimension mismatch: mask has {n_rec}, tensor has {L_rec_tensor}")
                    # Receptor 영역 (Intra)에 z_receptor 값 할당
                    z_unbound[b][rec_idx[:, None], rec_idx[None, :]] = z_receptor[b]

            else:
                # --- Bound 모드: 기존 단일 Complex 처리 ---
                s_init_complex, z_init_complex, _ = self.model.embed_input(batch_complex)
                _, s, z, _ = self.model.do_pairformer(
                    s_init_complex, z_init_complex, 
                    batch_complex['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    batch_complex['edge_mask'].shape[0]
                )
                z_complex = z


        num_batch = batch_complex['diffuse_mask'].shape[0]
        mutation = batch_complex['mutation']
        pdb_mt_id = batch_complex['processed_path'][0].split('/')[-1].replace('.pkl', '') + "_" + mutation
        if 'data_source' in batch_complex:
            data_source = batch_complex['data_source'][0]
            pdb_mt_id = batch_complex['processed_path'][0].split('/')[-1].replace('.pkl', '') + "_" + mutation + "_" + data_source
        diffuse_mask = batch_complex['diffuse_mask']
            
        sample_root_dir = os.path.join(self.inference_dir, pdb_mt_id)

        if not os.path.exists(sample_root_dir):
            os.makedirs(sample_root_dir, exist_ok=True)
            
        if pdb_mt_id != self.current_pdb_id:
            self.pairformer_cache.clear()
            self.current_pdb_id = pdb_mt_id

        atom37_traj, model_traj, pred_positions, pred_trans_1 = interpolant.sample(
            self.model,
            batch_complex,
            s_complex,
            s_complex,
            z_complex
        )

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1)) # (B, N_steps, L, 37, 3)
        pred_positions = du.to_numpy(pred_positions)
        pred_positions_37 = []

        batch_complex['residx_atom37_to_atom14'] = batch_complex['residx_atom37_to_atom14'][0]
        batch_complex['atom37_atom_exists'] = batch_complex['atom37_atom_exists'][0]

        b_factor_alt = diffuse_mask.cpu().numpy()
        b_factors = np.tile((b_factor_alt * 100)[:, :, None], (1, 1, 37)) # (B, L, 37)
        
        if hasattr(self, "affinity_model"):
            affinity_pred_value, affinity_pred_logit = self.affinity_model(
                s_inputs=s_init_complex[0],
                s_trunk=s_complex[0],
                z_bound=z_complex[0],
                z_unbound=z_unbound[0],
                loop_mask=batch_complex['loop_mask'][0],
                edge_mask=batch_complex['edge_mask'][0],
                inter_mask=inter_mask[0],
                x_pred_coords=pred_trans_1,
                )
            
        for i in range(pred_positions.shape[0]):
            pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch_complex) # (L_crop, 37, 3)
            pred_positions_37.append(pred_position_37)
        pred_positions = np.stack(pred_positions_37) # (B, L, 37, 3)

        samples = os.listdir(sample_root_dir)
        sample_nums = samples 
        next_sample_num = -1

        # protein의 n번째 (n>1) 배치를 생성할 때 
        if any('sample' in filename for filename in samples):
            sample_nums = sorted([int(sample.replace("sample_", "").replace(".pdb", "")) for sample in samples if ('sample' in sample) and not ('affinity' in sample)])
            next_sample_num = sample_nums[-1]

        for i in range(num_batch):
            next_sample_num += 1
            sample_path = os.path.join(sample_root_dir, f"sample_{next_sample_num}.pdb")
            pred_position = pred_positions[i]
            bb_traj = bb_trajs[i]

            # save structure data 
            aatype = du.to_numpy(batch_complex['aatype'][i].int())
            chain_idx = du.to_numpy(batch_complex['chain_idx'][i].int())
            residue_idx = du.to_numpy(batch_complex['residue_index'][i].int())
            diffuse_mask = du.to_numpy(batch_complex['diffuse_mask'][i].int())
            b_factor = b_factors[i]
            _ = eu.save_traj(
                sample=pred_position, # (L, 37, 3)
                bb_prot_traj=bb_traj, 
                x0_traj=np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                b_factors=b_factor,  # 위의 prmsd 집어넣기 
                diffuse_mask=diffuse_mask,
                output_path=sample_path,
                aatype=aatype,
                chain_index=chain_idx,
                residue_index=residue_idx,
                save_traj_bool=self._interpolant_cfg.save_traj
            )

            if hasattr(self, 'affinity_model'):
                affinity_json_path = os.path.join(sample_root_dir, f'sample_{next_sample_num}_affinity.json')
    
                affinity_dict = {
                    "category": batch_complex['mode'],
                    "affinity_pred_value": affinity_pred_value.squeeze().item(),
                    "affinity_pred_logit": affinity_pred_logit.squeeze().item(),
                    "affinity_true_log_value": torch.log10(torch.tensor(batch_complex['affinity_kd'][0])).squeeze().item()
                }
                with open(affinity_json_path, 'w') as f:
                    json.dump(affinity_dict, f, indent=4)