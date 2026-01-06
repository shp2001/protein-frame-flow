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

    def on_train_batch_start(self, batch, batch_idx):
        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        self.log(
            'lr',
            current_lr,
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
    # def on_train_batch_start(self, batch, batch_idx):
    #     # 모든 학습 가능한 파라미터 초기화
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
        
    #     if unused:
    #         print(f"🔥 실제로 사용되지 않은 파라미터: {unused}")
    #         raise RuntimeError("Unused parameters detected")  # 즉시 오류 발생시키기
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
        affinity_preds = []
        for i in range(2):
            batch = paired_batch[f"batch_{i}"]
            diffuse_mask = batch["diffuse_mask"]
            diffuse_mask_i = diffuse_mask[:, :, None]  # (B, L, 1)
            diffuse_mask_j = diffuse_mask[:, None, :]  # (B, 1, L)
            inter_mask = 1 - (diffuse_mask_i == diffuse_mask_j).float() # 0: diag / 1: off-diag

            # rollout for affinity
            self.rollout.set_device(batch['edge_mask'].device)
            with torch.no_grad():
                s_init, z_init, trans_perturbed = self.model.embed_input(batch)
                _, s, z, pair_outputs = self.model.do_pairformer(
                    s_init, 
                    z_init, 
                    batch['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    batch['edge_mask'].shape[0]
                    )
                
                if self._affinity_cfg.use_coords:
                    r3_traj, clean_atom37_traj, pred_positions, pred_trans_1 = self.rollout.sample(
                        self.model,
                        batch,
                        s,
                        s,
                        z
                    ) 
                else:
                    pred_trans_1 = None
            
            inter_pair_mask = batch['edge_mask'] * inter_mask
            affinity_pred = self.affinity_model(
                s_init[0],
                s[0],
                z[0],
                inter_pair_mask[0],
                pred_trans_1,
                )       
            affinity_preds.append(affinity_pred)
        
        # calculate affinity loss 
        affinity_pred_0 = affinity_preds[0]
        affinity_pred_1 = affinity_preds[1]
        target = (1.0 - 2.0 * paired_batch['label']).to(affinity_pred_0.device)

        margin_loss_fn = nn.MarginRankingLoss(margin=0.1, reduce=False)
        affinity_loss = margin_loss_fn(affinity_pred_0, affinity_pred_1, target)      # batch_0 > batch_1 * 10 -> label: 0
        if len(affinity_loss.shape) == 1:
            affinity_loss = affinity_loss[None, ...]
        
        total_loss = affinity_loss
        
        return {
            "total_loss": total_loss,
            "affinity_loss": affinity_loss,
        }


    def validation_step(self, paired_batch: Any, batch_idx: int):

        affinity_preds = []
        for i in range(2):
            batch = paired_batch[f"batch_{i}"]
            loop_mask = batch['loop_mask']
            self.interpolant.set_device(loop_mask.device)
            num_batch, num_res = loop_mask.shape

            raw_path = batch['raw_path']
            print("raw_path", raw_path)
            pdb_id = raw_path.split('/')[-1].replace('.pdb', '')

            diffuse_mask = batch["diffuse_mask"]
            diffuse_mask_i = diffuse_mask[:, :, None]  # (B, L, 1)
            diffuse_mask_j = diffuse_mask[:, None, :]  # (B, 1, L)
            inter_mask = 1 - (diffuse_mask_i == diffuse_mask_j).float() # 0: diag / 1: off-diag

            # rollout for affinity
            with torch.no_grad():
                s_init, z_init, trans_perturbed = self.model.embed_input(batch)
                _, s, z, pair_outputs = self.model.do_pairformer(
                    s_init, 
                    z_init, 
                    batch['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    batch['edge_mask'].shape[0]
                    )
                
                if self._affinity_cfg.use_coords:
                    r3_traj, clean_atom37_traj, pred_positions, pred_trans_1 = self.interpolant.sample(
                        self.model,
                        batch,
                        s,
                        s,
                        z
                    ) 
                else:
                    pred_trans_1 = None

            # if hasattr(self, "confidence_model"):
            #     plddt_pred, pae_pred = self.confidence_model(
            #         batch['ref_feature_dict'],
            #         s_init[0],
            #         s[0],
            #         z[0],
            #         batch['edge_mask'][0],
            #         pred_trans_1,
            #     )  

            # save 3d structure for validation 
            pred_positions_37 = []
            pred_positions = du.to_numpy(pred_positions)
            for i in range(pred_positions.shape[0]):
                pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch)
                pred_positions_37.append(pred_position_37)
            
            pred_positions = np.stack(pred_positions_37)
            batch_metrics = []
            
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'{pdb_id}_len_{num_res}'
            )
            os.makedirs(sample_dir, exist_ok=True)

                
            for i in range(num_batch):
                # Write out sample to PDB file (wo b-factors)
                final_pos = pred_positions[i]

                if batch['mode'] == 'polymer' or batch['mode'] == 'monomer':
                    unique_vals, mapped = torch.unique(batch['chain_idx'][0], return_inverse=True)
                    batch['chain_idx'] = mapped.unsqueeze(0).expand(num_batch, -1)

                saved_path = au.write_prot_to_pdb(
                    final_pos,
                    file_path=os.path.join(sample_dir, pdb_id+'.pdb'),
                    aatype=batch['aatype'].cpu(),
                    chain_index=batch['chain_idx'].cpu(),
                    no_indexing=False,
                    overwrite=True,
                    b_factors=None
                )

            # affinity prediction 
            inter_pair_mask = batch['edge_mask'] * inter_mask
            affinity_pred = self.affinity_model(
                s_init[0],
                s[0],
                z[0],
                inter_pair_mask[0],
                pred_trans_1,
                )       
            affinity_preds.append(affinity_pred)

        # calculate affinity loss 
        affinity_pred_0 = affinity_preds[0]
        affinity_pred_1 = affinity_preds[1]
        target = (1.0 - 2.0 * paired_batch['label']).to(affinity_pred_0.device)

        margin_loss_fn = nn.MarginRankingLoss(margin=0.1, reduce=False)
        affinity_loss = margin_loss_fn(affinity_pred_0, affinity_pred_1, target)      # batch_0 > batch_1 * 10 -> label: 0
        if len(affinity_loss.shape) == 1:
            affinity_loss = affinity_loss[None, ...]
        affinity_loss_dict = {'affinity_loss': affinity_loss}
        batch_metrics.append(affinity_loss_dict)

        # calculate trans diffuse loss (rmsd)
        gt_trans_1 = batch['trans_1']
        trans_error = (gt_trans_1 - pred_trans_1) 
        trans_diffuse_loss = torch.sum(
            trans_error ** 2 * diffuse_mask[..., None],
            dim=(-1, -2)
        ) / (torch.sum(diffuse_mask, dim=-1) * 3)
        trans_diffuse_loss_dict = {'trans_diffuse_loss': trans_diffuse_loss**0.5}
        batch_metrics.append(trans_diffuse_loss_dict)

        frame_mask = diffuse_mask * (1 - loop_mask)
        trans_frame_loss = torch.sum(
            trans_error ** 2 * frame_mask[..., None],
            dim=(-1, -2)
        ) / (torch.sum(diffuse_mask, dim=-1) * 3)
        trans_diffuse_loss_dict = {'trans_frame_loss': trans_frame_loss**0.5}
        batch_metrics.append(trans_diffuse_loss_dict)

        # calculate trans loop loss (rmsd)
        trans_loop_loss = torch.sum(
            trans_error ** 2 * loop_mask[..., None],
            dim=(-1, -2)
        ) / (torch.sum(loop_mask, dim=-1) * 3)
        trans_loop_loss_dict = {'trans_loop_loss': trans_loop_loss**0.5}
        batch_metrics.append(trans_loop_loss_dict)

        # calcuclate trans loss (h3 rmsd)
        h3_mask = torch.zeros_like(loop_mask)
        count = 0
        for i in range(num_batch):
            count = 0  
            in_group = False  
            group_start = None  
            
            # 연속된 1들의 그룹을 추적
            for j in range(num_res):
                if loop_mask[i, j] == 1:
                    if not in_group:  # 새로운 그룹 시작
                        group_start = j
                        in_group = True
                else:
                    if in_group:  # 그룹이 끝나는 지점
                        count += 1
                        # 세 번째 그룹만 남기고 나머지 그룹은 0
                        if count == 3:
                            h3_mask[i, group_start:j] = 1
                        in_group = False
            
        # calculate h3 trans loss (rmsd)
        h3_trans_loss = torch.sum(
            trans_error ** 2 * h3_mask[..., None],
            dim=(-1, -2)
        ) / (torch.sum(h3_mask, dim=-1) * 3)
        h3_trans_loss_dict = {'h3_trans_loss': h3_trans_loss**0.5}

        batch_metrics.append(h3_trans_loss_dict)

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
        
        num_batch = batch_losses['total_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs):
    
        if self.is_loss_nan:
            self.print(f"❌ NaN in closure at step {self.global_step}")
            optimizer.zero_grad()
            return

        # grad 검사
        nan_in_grad = False
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                self.print(f"⚠️ NaN in grad: {name}")
                nan_in_grad = True

        if nan_in_grad:
            optimizer.zero_grad()
            return

        optimizer.step(closure=optimizer_closure)

    def on_train_batch_start(self, batch, batch_idx):
        # 첫 번째 optimizer 기준
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        print(f"[Step {self.global_step}] Learning Rate: {lr:.6f}")

    def on_predict_start(self):
        self.pairformer_cache = {}
        self.current_pdb_id = None
        print("Pairformer cache has been initialized.")

    def predict_step(self, batch, batch_idx):
        del batch_idx # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        num_batch = batch['sample_id'].shape[0]
        pdb_id = batch['processed_path'].split('/')[-1].replace('.pkl', '')

        if 'sample' in pdb_id:
            parts = batch['processed_path'].split('/')
            pdb_id = parts[-2] + "_" + parts[-1].replace('.pkl', '')
        sample_root_dir = os.path.join(self.inference_dir, pdb_id)
        if not os.path.exists(sample_root_dir):
            os.makedirs(sample_root_dir, exist_ok=True)
            
        

# ############################################################################################################
        if pdb_id != self.current_pdb_id:
            self.pairformer_cache.clear()
            self.current_pdb_id = pdb_id

        if pdb_id in self.pairformer_cache:
            s_init, s, z, trans_perturbed = self.pairformer_cache[pdb_id]
            
        else:
            s_init_embed, z_init, trans_perturbed = self.model.embed_input(batch)
            s_init, s, z, pair_outputs = self.model.do_pairformer(
                s_init_embed,
                z_init,
                batch['edge_mask'][0][None, ...],
                self._model_cfg.pairformer.n_cycles,
                num_batch
            )

            # 결과를 캐시에 저장합니다.
            self.pairformer_cache[pdb_id] = (s_init, s, z, trans_perturbed)
############################################################################################################
############################################################################################################
        # s_init_embed, z_init, trans_perturbed = self.model.embed_input(batch)
        # s_init, s, z, pair_outputs = self.model.do_pairformer(
        #     s_init_embed,
        #     z_init,
        #     batch['edge_mask'][0][None, ...],
        #     self._model_cfg.pairformer.n_cycles,
        #     num_batch
        # )

        # # row-wise
        # loop_mask = batch['loop_mask'][0]
        # rows_selected = z[0][loop_mask, :, :]          # [L_selected, L, C]
        # # col-wise
        # cols_selected = z[0][:, loop_mask, :]          # [L, L_selected, C]
        # cols_selected = cols_selected.permute(1, 0, 2)     # [L_selected, L, C]
        # # concat
        # z_cdr = torch.cat([rows_selected, cols_selected], dim=0)  # [L_selected*2, L, C]

        # # flatten to [feature_dim]
        # z_cdr = z_cdr.mean(dim=1).mean(dim=0)  # [C]
        # torch.save({"z": z[0], "loop_mask": loop_mask}, os.path.join(sample_root_dir, "pair.pt"))

############################################################################################################

        atom37_traj, model_traj, pred_positions, pred_trans_1 = interpolant.sample(
            self.model,
            batch,
            s_init,
            s,
            z
        )

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1)) # (B, N_steps, L, 37, 3)
        pred_positions = du.to_numpy(pred_positions)
        pred_positions_37 = []

        batch['residx_atom37_to_atom14'] = batch['residx_atom37_to_atom14'][0]
        batch['atom37_atom_exists'] = batch['atom37_atom_exists'][0]


        if hasattr(self, "confidence_model"):
            plddt_pred, pae_pred = self.confidence_model(
                batch['ref_feature_dict'],
                s_init[0],
                s[0],
                z[0],
                batch['edge_mask'][0],
                pred_trans_1,
            )       
            plddt_bins = plddt_pred.shape[-1]
            plddt_probs = nn.functional.softmax(plddt_pred, dim=-1)  # [B, N_atom, plddt_bins]
            bin_values = torch.linspace(0, 1, plddt_bins, device=plddt_pred.device)  # [plddt_bins]
            plddt_score = torch.sum(plddt_probs * bin_values, dim=-1)  # [B, N_atom]
            b_factors_14 = du.atom_unflatten(plddt_score, batch['atom14_gt_exists']) # [B, N_token, 14]
            b_factors_14 = du.to_numpy(b_factors_14) * 100


            # cdr 별 plddt 저장 
            anchors = find_anchor(batch['loop_mask'][0], only_h3=False)
            plddts_by_cdr_all_atom = []
            plddts_by_cdr_backbone = []
            for i in range(len(anchors)//2):
                start = anchors[2*i]
                end = anchors[2*i+1]

                # all atom plddt 
                cdr_b_factors = b_factors_14[:, start+1:end, :] # (B, L_cdr, 14)
                cdr_atom14_mask = du.to_numpy(batch['atom14_gt_exists'][:, start+1:end, :]) # (B, L_cdr, 14)
                plddts_by_cdr_all_atom.append(np.sum(cdr_b_factors, axis=(-1, -2)) / np.sum(cdr_atom14_mask, axis=(-1, -2))) # (B)

                # all atom plddt 
                cdr_b_factors_bb = b_factors_14[:, start+1:end, :3] # (B, L_cdr, 3)
                cdr_atom14_mask_bb = du.to_numpy(batch['atom14_gt_exists'][:, start+1:end, :3]) # (B, L_cdr, 3)
                plddts_by_cdr_backbone.append(np.sum(cdr_b_factors_bb, axis=(-1, -2)) / np.sum(cdr_atom14_mask_bb, axis=(-1, -2))) # (B)
                
            plddts_by_cdr_all_atom = np.stack(plddts_by_cdr_all_atom, axis=1) # (B, 6)
            plddts_by_cdr_backbone = np.stack(plddts_by_cdr_backbone, axis=1) # (B, 6)

            # b_factor 차원 (14 -> 37)
            b_factors = []
            for i in range(pred_positions.shape[0]):
                b_factors_37 = all_atom.atom14_to_atom37(b_factors_14[i][..., None], batch)
                b_factors.append(np.squeeze(b_factors_37, axis=-1))
            b_factors = np.stack(b_factors) # (B, L, 37)


        else:
            b_factor_alt = diffuse_mask.cpu().numpy()
            b_factors = np.tile((b_factor_alt * 100)[:, :, None], (1, 1, 37)) # (B, L, 37)

        for i in range(pred_positions.shape[0]):
            pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch) # (L_crop, 37, 3)
            pred_positions_37.append(pred_position_37)
        pred_positions = np.stack(pred_positions_37) # (B, L, 37, 3)

        samples = os.listdir(sample_root_dir)
        sample_nums = samples 
        next_sample_num = -1

        # protein의 n번째 (n>1) 배치를 생성할 때 
        if any('sample' in filename for filename in samples):
            sample_nums = sorted([int(sample.replace("sample_", "").replace(".pdb", "")) for sample in samples if ('sample' in sample) and not ('plddt' in sample)])
            next_sample_num = sample_nums[-1]

        for i in range(num_batch):
            next_sample_num += 1
            sample_path = os.path.join(sample_root_dir, f"sample_{next_sample_num}.pdb")
            pred_position = pred_positions[i]
            bb_traj = bb_trajs[i]

            # save structure data 
            aatype = du.to_numpy(batch['aatype'][i].int())
            chain_idx = du.to_numpy(batch['chain_idx'][i].int())
            residue_idx = du.to_numpy(batch['residue_index'][i].int())
            diffuse_mask = du.to_numpy(batch['diffuse_mask'][i].int())
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
            # save perturbed trans 
            if trans_perturbed != None:
                perturbed_trans_path = os.path.join(sample_root_dir, f"sample_{next_sample_num}_perturbed_trans.pdb")
                du.save_perturbed_trans(
                    trans_perturbed[i][None, ...], 
                    batch['chain_index'][i][None, ...], 
                    batch['residue_index'][i][None, ...],
                    perturbed_trans_path
                    )

            # save plddt 
            if hasattr(self, "confidence_model"):
                plddt_by_cdr_all_atom = plddts_by_cdr_all_atom[i].tolist()
                plddt_by_cdr_backbone = plddts_by_cdr_backbone[i].tolist()
                plddt_by_cdr_dict = {
                    'h1_aa': plddt_by_cdr_all_atom[0],
                    'h2_aa': plddt_by_cdr_all_atom[1],
                    'h3_aa': plddt_by_cdr_all_atom[2],
                    'l1_aa': plddt_by_cdr_all_atom[3],
                    'l2_aa': plddt_by_cdr_all_atom[4],
                    'l3_aa': plddt_by_cdr_all_atom[5],       
                    'h1_bb': plddt_by_cdr_backbone[0],
                    'h2_bb': plddt_by_cdr_backbone[1],
                    'h3_bb': plddt_by_cdr_backbone[2],
                    'l1_bb': plddt_by_cdr_backbone[3],
                    'l2_bb': plddt_by_cdr_backbone[4],
                    'l3_bb': plddt_by_cdr_backbone[5],          
                }
                
                # 저장할 파일 경로
                json_path = os.path.join(sample_root_dir, f'sample_{next_sample_num}_plddts.json')

                # JSON으로 저장
                with open(json_path, 'w') as f:
                    json.dump(plddt_by_cdr_dict, f, indent=4)