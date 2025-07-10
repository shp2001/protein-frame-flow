from typing import Any
import torch
import time
import math 

import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model import FlowModel, ConfidenceModel
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger
from models.loss import *

class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)
        self.confidence_model = None
        if cfg.model.prmsd.use_prmsd:
            self.confidence_model = ConfidenceModel(cfg.model)
        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)
        self.mini_rollout = Interpolant(cfg.mini_rollout)
        self.learning_rate = self._exp_cfg.optimizer.max_lr
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None
        self.save_file = True

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
        # 모든 학습 가능한 파라미터 초기화
        for p in self.parameters():
            if p.requires_grad:
                p.grad = None
        
        # Forward pass 전 파라미터 기록 (메모리 주소까지 추적)
        self._params_before = {id(p): n for n, p in self.named_parameters() if p.requires_grad}

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

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape
        device = noisy_batch['trans_1'].device 

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_chi_angle = noisy_batch['chi_angles_sin_cos']
        gt_atom14_pos = noisy_batch['atom14_gt_positions'].clone()
        alt_atom14_pos = noisy_batch['atom14_alt_gt_positions'].clone()
        gt_pseudo_beta = noisy_batch['pseudo_beta'].clone()
        backbone_rigid_tensor = noisy_batch['backbone_rigid_tensor'].clone()
        rigidgroups_gt_frames = noisy_batch['rigidgroups_gt_frames'].clone()
        rigidgroups_alt_gt_frames = noisy_batch['rigidgroups_alt_gt_frames'].clone()

        print("raw_path", noisy_batch['raw_path'])
        
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        # if torch.any(torch.isnan(gt_rot_vf)):
        #     raise ValueError('NaN encountered in gt_rot_vf')

        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t']
        so3_t = noisy_batch['so3_t']
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        gt_atom14_pos = gt_atom14_pos * training_cfg.bb_atom_scale / r3_norm_scale[..., None] # scaling 
        gt_bb_atoms = gt_atom14_pos[:, :, :3] # scaling 

        alt_atom14_pos = alt_atom14_pos * training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        gt_pseudo_beta = gt_pseudo_beta * training_cfg.bb_atom_scale / r3_norm_scale

        backbone_rigid_tensor[..., :3, 3] = backbone_rigid_tensor[..., :3, 3] * training_cfg.bb_atom_scale / r3_norm_scale
        rigidgroups_gt_frames[..., :3, 3] = rigidgroups_gt_frames[..., :3, 3] * training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        rigidgroups_alt_gt_frames[..., :3, 3] = rigidgroups_alt_gt_frames[..., :3, 3] * training_cfg.bb_atom_scale / r3_norm_scale[..., None]

        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans'].clone()
        pred_rotmats_1 = model_output['pred_rotmats'].clone()
        pred_atom_14_list = model_output['all_atom_preds']['positions'].clone()
        # pred_angles_list = model_output['all_atom_preds']['angles'].clone()
        # pred_unnormalized_angles_list = model_output['all_atom_preds']['unnormalized_angles'].clone()

        pred_atom_14_list = [pred * training_cfg.bb_atom_scale / r3_norm_scale[..., None] for pred in pred_atom_14_list]
        pred_atom_14_list = torch.stack(pred_atom_14_list, dim=0) # (O, B, L, A, 3)

        if torch.isnan(pred_atom_14_list).any():
            raise ValueError(f"pred_aa_contact_map: {torch.isnan(pred_atom_14_list).any()} \n")
        
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        # if torch.any(torch.isnan(pred_rots_vf)):
        #     raise ValueError('NaN encountered in pred_rots_vf')

        # Get the renamed ground truth 
        renamed_dict = compute_renamed_ground_truth(noisy_batch,
                                                    atom14_pred_positions=model_output['all_atom_preds']["positions"][-1])

        alt_naming_is_better = renamed_dict['alt_naming_is_better'].clone()
        renamed_atom14_gt_exists = renamed_dict['renamed_atom14_gt_exists'].clone()
        renamed_atom14_gt_positions = renamed_dict['renamed_atom14_gt_positions'] * training_cfg.bb_atom_scale / r3_norm_scale[..., None]

        # Translation VF loss
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # local Pairwise distance loss (final layer만 계산)
        local_dist_mat_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
        scale_factor = training_cfg.bb_atom_scale / (1 - torch.min(
        r3_t, torch.tensor(training_cfg.t_normalize_clip)))

        if training_cfg.aux_loss_use_local_dist_mat_loss:
            pred_atom_14 = pred_atom_14_list[-1]
            local_dist_mat_loss, neighbor_indices, cdr_residues = local_distance_loss(
                pred_atom_14, # scaled 
                renamed_atom14_gt_exists,
                renamed_atom14_gt_positions, # scaled 
                noisy_batch['original_diffuse_mask'][0],
                scale_factor,
                noisy_batch['mode']
            )   # local_loss_mask: (B, N, N, 14)

        # Backbone atom loss
        bb_atom_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
        if training_cfg.aux_loss_use_bb_loss:
            bb_atom_loss = compute_rmsd(pred_atom_14_list,
                                    renamed_atom14_gt_positions,
                                    cdr_mask=noisy_batch['diffuse_mask'],
                                    atom14_gt_exists=renamed_atom14_gt_exists,
                                    mode='bb',
                                    data_mode=noisy_batch['mode'],
                                    compute_non_cdr=False,
                                    compute_cdr=True,
                                    compute_h3=True
                                    )
                                
        # sc atom loss (final layer만 계산)
        sc_atom_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)

        interface_mask = noisy_batch['diffuse_mask'].clone()
        interface_mask[:, neighbor_indices] = 1

        if training_cfg.aux_loss_use_sc_atom_loss:
            sc_atom_loss = compute_rmsd(pred_atom_14_list,
                                    renamed_atom14_gt_positions,
                                    cdr_mask=interface_mask,
                                    atom14_gt_exists=renamed_atom14_gt_exists,
                                    mode='sc',
                                    data_mode=noisy_batch['mode'],
                                    compute_non_cdr=True,
                                    compute_cdr=True,
                                    compute_h3=False
                                    )    
        # torsion angle loss 
        # chi_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        # if training_cfg.aux_loss_use_chi_loss:
        #     chi_loss = supervised_chi_loss(pred_angles_list,
        #                                 pred_unnormalized_angles_list,
        #                                 noisy_batch['aatype'],
        #                                 noisy_batch['res_mask'],
        #                                 noisy_batch['chi_mask'],
        #                                 gt_chi_angle,
        #                                 chi_weight=0.5,
        #                                 angle_norm_weight=0.02,
        #                                 cdr_mask=interface_mask
        #                                 )
            
        # final layer backbone rmsd loss
        final_bb_rmsd = compute_rmsd(pred_atom_14_list[-1].unsqueeze(0),
                                renamed_atom14_gt_positions,
                                cdr_mask=noisy_batch['diffuse_mask'],
                                atom14_gt_exists=renamed_atom14_gt_exists,
                                mode='bb',
                                data_mode=noisy_batch['mode'],
                                compute_non_cdr=False,
                                compute_cdr=True,
                                compute_h3=True
                                )

        final_layer_rmsd = final_bb_rmsd * (training_cfg.aux_loss_bb_atom_loss_weight/2)
        

        # all atom clash loss 
        batch_size = gt_atom14_pos.shape[0]
        all_atom_clash_loss = torch.zeros(batch_size, device=device)
        if training_cfg.aux_loss_use_all_atom_clash_loss:
            try:
                all_atom_clash_loss = compute_all_atom_clash_loss(
                    model_output['all_atom_preds']['positions'][-1],
                    noisy_batch['atom14_gt_exists'],
                    noisy_batch['res_idx'],
                    noisy_batch['residx_atom14_to_atom37'],
                    interface_mask=interface_mask
                )
            except Exception as e:
                print(f"[Warning] all atom clash loss skipped due to error: {e}")
                all_atom_clash_loss = torch.zeros(batch_size).to(device)

        within_clash_loss = torch.zeros(batch_size, device=device)
        if training_cfg.aux_loss_use_within_clash_loss:
            try:
                within_clash_loss = compute_within_clash_loss(
                                                            model_output['all_atom_preds']['positions'][-1],
                                                            noisy_batch['atom14_gt_exists'],
                                                            interface_mask,
                                                            noisy_batch['aatype'])   
            except Exception as e:
                print(f"[Warning] within clash loss skipped due to error: {e}")
                within_clash_loss = torch.zeros(batch_size).to(device)


        # calculate prmsd (perform mini rollout with 10 timesteps)
        prmsd_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
        if training_cfg.aux_loss_use_prmsd_loss:
            self.mini_rollout.set_device(loss_mask.device)

            cdr_residues, interface_residues = au.get_cdr_and_neighbors(
                atom14_gt_positions=noisy_batch["atom14_gt_positions"],
                atom14_gt_exists=noisy_batch["atom14_gt_exists"],
                original_diffuse_mask=noisy_batch["original_diffuse_mask"][0],
                mode=noisy_batch['mode'],
                scale_factor=torch.ones(num_batch)
                )
            
            interface_mask = noisy_batch['diffuse_mask'].clone()
            interface_mask[:, interface_residues] = 1
            noisy_batch['interface_mask'] = interface_mask

            _, _, mini_pred_positions, prmsd_final, mini_prmsd, _, _, input_for_confidence = self.mini_rollout.sample(
                num_batch,
                num_res,
                self.model,
                noisy_batch,
                rollout=True
            )

            mini_prmsd = self.confidence_model(input_for_confidence, noisy_batch['res_mask'])
            prmsd_loss = lddt_loss(logits=mini_prmsd,
                                    all_atom_pred_pos=mini_pred_positions, # predicted structure (b, l, 14, 3)
                                    all_atom_positions=renamed_dict["renamed_atom14_gt_positions"], # gt stucture  (b, l, 14, 3)
                                    all_atom_mask=renamed_dict["renamed_atom14_gt_exists"],
                                    cdr_mask=noisy_batch['diffuse_mask']) # (b, l)

            # final_prmsd = compute_prmsd(pred_rmsd, cdr_mask=noisy_batch['diffuse_mask'])
            # print(f"prmsd_max: {torch.max(final_prmsd[0])}")

        # calculate auxiliary loss 
        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss * training_cfg.aux_loss_bb_atom_loss_weight
            + sc_atom_loss * training_cfg.aux_loss_use_sc_atom_loss * training_cfg.aux_loss_sc_atom_loss_weight
            + local_dist_mat_loss * training_cfg.aux_loss_use_local_dist_mat_loss * training_cfg.aux_loss_local_dist_mat_loss_weight
            + final_layer_rmsd * training_cfg.aux_loss_use_final_layer_rmsd * training_cfg.aux_loss_final_layer_rmsd_weight
        )

        # calculate violation loss
        violation_loss = (
            all_atom_clash_loss * training_cfg.aux_loss_use_all_atom_clash_loss * training_cfg.aux_loss_all_atom_clash_loss_weight
            + within_clash_loss * training_cfg.aux_loss_use_within_clash_loss * training_cfg.aux_loss_within_clash_loss_weight
        )
        auxiliary_loss *= (
            (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=18)

        violation_loss *= (
            (r3_t[:, 0] > training_cfg.viol_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.viol_loss_t_pass)
        )
    
        se3_vf_loss = se3_vf_loss + auxiliary_loss + violation_loss + prmsd_loss * training_cfg.aux_loss_prmsd_loss_weight
        if torch.any(torch.isnan(se3_vf_loss)):
            se3_vf_loss = torch.nan_to_num(se3_vf_loss, nan=0.0)
            
        # print({
        #     "r3_t": r3_t,
        #     "trans_loss": trans_loss,
        #     "bb_atom_loss": bb_atom_loss,
        #     'sc_atom_loss': sc_atom_loss,
        #     'all_atom_clash_loss': all_atom_clash_loss,
        #     'within_clash_loss': within_clash_loss,
        #     'local_dist_mat_loss': local_dist_mat_loss,
        #     'distogram_loss': distogram_loss,
        #     'contact_map_loss': contact_map_loss,
        #     'prmsd_loss': prmsd_loss
        # })

        return {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "bb_atom_loss": bb_atom_loss,
            'sc_atom_loss': sc_atom_loss,
            'all_atom_clash_loss': all_atom_clash_loss,
            'within_clash_loss': within_clash_loss,
            'local_dist_mat_loss': local_dist_mat_loss,
            'prmsd_loss': prmsd_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        b = res_mask.shape[0]
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch['diffuse_mask']
        raw_path = batch['raw_path']
        pdb_id = raw_path.split('/')[-1].replace('.pdb', '')

        atom37_traj, clean_atom37_traj, pred_positions, prmsd_final, prmsd, pred_trans_1, pred_rotmats_1, input_for_confidence = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            batch
        )
        
        pred_positions_37 = []
        pred_positions = du.to_numpy(pred_positions)
        for i in range(pred_positions.shape[0]):
            pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch)
            pred_positions_37.append(pred_position_37)
        
        pred_positions = np.stack(pred_positions_37)
        batch_metrics = []

        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'{pdb_id}_len_{num_res}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file (wo b-factors)
            final_pos = pred_positions[i]
            b_factors = prmsd_final[i].cpu().numpy()
            if (b_factors==0).all():
                b_factor_alt = diffuse_mask.cpu().numpy()
                b_factors = np.tile((b_factor_alt[i] * 100)[:, None], (1, 37))
            
            else:
                b_factors = np.tile((b_factors)[:, None], (1, 37))

            saved_path = au.write_prot_to_pdb(
                final_pos,
                file_path=os.path.join(sample_dir, pdb_id+'.pbd'),
                aatype=batch['aatype'].cpu(),
                chain_index=batch['chain_idx'].cpu(),
                no_indexing=False,
                overwrite=True,
                b_factors=b_factors
            )
            
            # print(f'contact_map: {contact_map[i].shape}')
            # print(f'gt_cb_contact_map: {gt_cb_contact_map[i].shape}')

            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order['CA']
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

            # calculate trans loss (rmsd)
            gt_trans_1 = batch['trans_1']
            trans_error = (gt_trans_1 - pred_trans_1) 
            trans_loss = torch.sum(
                trans_error ** 2 * diffuse_mask[..., None],
                dim=(-1, -2)
            ) / (torch.sum(diffuse_mask, dim=-1) * 3)
            trans_loss_dict = {'trans_loss': trans_loss**0.5}
            batch_metrics.append(trans_loss_dict)

            # calcuclate trans loss (h3 rmsd)
            b, N = diffuse_mask.shape
            h3_mask = torch.zeros_like(diffuse_mask)
            count = 0
            for i in range(b):
                count = 0  
                in_group = False  
                group_start = None  
                
                # 연속된 1들의 그룹을 추적
                for j in range(N):
                    if diffuse_mask[i, j] == 1:
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
            )
        self.validation_epoch_metrics.clear()

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
            on_epoch=False,
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


    
    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        params_before = {n: p.requires_grad for n, p in self.named_parameters() if p.requires_grad}
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
                noisy_batch['rotmats_sc'] = (
                    model_sc['pred_rotmats'] * noisy_batch['diffuse_mask'][..., None, None]
                    + noisy_batch['rotmats_1'] * (1 - noisy_batch['diffuse_mask'][..., None, None])
                )
        try:
            batch_losses = self.model_step(noisy_batch)
        except Exception as e:
            print(f"Error during model_step: {e}")
            zero_loss = torch.zeros(batch['res_mask'].shape[0], device=batch['res_mask'].device)
            batch_losses = {
            "trans_loss": zero_loss,
            "auxiliary_loss": zero_loss,
            "rots_vf_loss": zero_loss,
            "se3_vf_loss": zero_loss,
            "bb_atom_loss": zero_loss,
            'sc_atom_loss': zero_loss,
            'all_atom_clash_loss': zero_loss,
            'within_clash_loss': zero_loss,
            'local_dist_mat_loss': zero_loss,
            'prmsd_loss': zero_loss,
            'distogram_loss': zero_loss,
            'contact_map_loss': zero_loss
        }

        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            else:
                batch_t = r3_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        scaffold_percent = torch.mean(batch['diffuse_mask'].float()).item()
        self._log_scalar(
            "train/scaffolding_percent",
            scaffold_percent, prog_bar=False, batch_size=num_batch)
        motif_mask = 1 - batch['diffuse_mask'].float()
        num_motif_res = torch.sum(motif_mask, dim=-1)
        self._log_scalar(
            "train/motif_size", 
            torch.mean(num_motif_res).item(), prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses['se3_vf_loss']
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
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
            else:
                theta = (step - warmup_steps) / (total_steps - warmup_steps) * math.pi
                cosine_decay = 0.5 * (1 + math.cos(theta))
                return (min_lr / max_lr) + cosine_decay * (1 - (min_lr / max_lr))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, steplr_with_warmup)
    

    def configure_optimizers(self):
        if self.confidence_model != None:
            parameters = list(self.model.parameters()) + list(self.confidence_model.parameters())
        else:
            parameters = self.model.parameters()
        optimizer = torch.optim.AdamW(
            parameters, self.learning_rate, weight_decay=0.01
        )
        # scheduler = self.get_cosine_scheduler_w_warmup(
        #     optimizer,
        #     self._exp_cfg.optimizer.warmup_steps,
        #     self._exp_cfg.optimizer.decay_steps,
        #     self._exp_cfg.optimizer.min_lr,
        #     self._exp_cfg.optimizer.max_lr,
        # )
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def on_train_batch_start(self, batch, batch_idx):
        # 첫 번째 optimizer 기준
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        print(f"[Step {self.global_step}] Learning Rate: {lr:.6f}")
        
    def predict_step(self, batch, batch_idx):
        del batch_idx # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_ids = batch['sample_id'].squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)


        pdb_id = batch['raw_path'].split('/')[-1].replace('.pdb', '')
        print("pdb_id", pdb_id)
        if 'diffuse_mask' in batch: # motif-scaffolding
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            diffuse_mask = batch['diffuse_mask']

            true_bb_pos = all_atom.atom37_from_trans_rot(trans_1, rotmats_1, 1 - diffuse_mask)
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            _, sample_length, _ = trans_1.shape

        else: # unconditional
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            trans_1 = rotmats_1 = diffuse_mask = None
            diffuse_mask = torch.ones(1, sample_length, device=device)

        # Sample batch
        if self.save_file:
            sample_dirs = [os.path.join(
                self.inference_dir, pdb_id, f'sample_{sample_id}')
                for sample_id in sample_ids]

            atom37_traj, model_traj, pred_positions, prmsd_final, prmsd, pred_trans_1, pred_rotmats_1, input_for_confidence = interpolant.sample(
                num_batch, 
                sample_length, 
                self.model,
                batch
            )

            if self.confidence_model != None:
                prmsd = self.confidence_model(input_for_confidence, batch['res_mask'])
                prmsd_final = compute_plddt(prmsd, cdr_mask=batch['diffuse_mask'])
            # cdr_residues, neighbor_indices = au.get_cdr_and_neighbors(
            #     torch.tensor(pred_positions, device=batch['aatype'].device),
            #     batch['atom14_gt_positions'],
            #     batch['atom14_gt_exists'],
            #     batch['original_diffuse_mask'][0],
            #     batch['mode'],
            #     scale_factor=torch.ones(b),
            #     distance_threshold=5
            # )
            
            bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
            pred_positions_37 = []

            pred_positions = du.to_numpy(pred_positions) # (B, L_crop, 14 , 3)
            gt_positions = du.to_numpy(batch['original_atom14_gt_positions']) # (L, 14, 3)

            for i in range(pred_positions.shape[0]):
                pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch) # (L_crop, 37, 3)
                gt_batch = {
                    'residx_atom37_to_atom14': batch['original_residx_atom37_to_atom14'],
                    'atom37_atom_exists': batch['original_atom37_atom_exists']
                }
                gt_position_37 = all_atom.atom14_to_atom37(gt_positions, gt_batch) # (L, 37, 3)
                gt_position_37[batch['res_idx'][i].cpu().numpy()] = pred_position_37
                pred_positions_37.append(gt_position_37)


            pred_positions = np.stack(pred_positions_37)

            total_prmsd = torch.zeros(pred_positions.shape[0], gt_positions.shape[0], device=batch['res_idx'].device)
            total_prmsd.scatter_(dim=1, index=batch['res_idx'], src=prmsd_final)
            prmsds = du.to_numpy(total_prmsd)

            for i in range(num_batch):
                sample_dir = sample_dirs[i]
                pred_position = pred_positions[i]
                prmsd = prmsds[i]
                bb_traj = bb_trajs[i]
                os.makedirs(sample_dir, exist_ok=True)
                aatype = du.to_numpy(batch['original_aatype'].long())
                chain_idx = du.to_numpy(batch['original_chain_idx'].long())

                # au.visualize_contact_map(contact_map[i, :, :, 50] * cb_mask[i], cdr_residues, neighbor_indices,
                #                          title=pdb_id.split('_')[0] + '_pred',
                #                          output_path=os.path.join(sample_dir, 'pred_contact_map.png'))
                # au.visualize_contact_map(gt_cb_contact_map[i], cdr_residues, neighbor_indices,
                #                          title=pdb_id.split('_')[0] + '_gt',
                #                          output_path=os.path.join(sample_dir, 'gt_contact_map.png'))

                print("original_diffuse_mask", batch['original_diffuse_mask'].shape)
                _ = eu.save_traj(
                    sample=pred_position, # (L, 37, 3)
                    bb_prot_traj=bb_traj, 
                    x0_traj=np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                    b_factors=prmsd,  # 위의 prmsd 집어넣기 
                    diffuse_mask=batch['original_diffuse_mask'].cpu().numpy(),
                    output_dir=sample_dir,
                    aatype=aatype,
                    chain_index=chain_idx,
                    save_traj_bool=False
                )

        else:
            if not os.path.exists('/home/psh/protein-frame-flow/train_conf'):
                os.makedirs('/home/psh/protein-frame-flow/train_conf', exist_ok=True)

            sample_files = [os.path.join(
                '/home/psh/protein-frame-flow/train_conf', f'{pdb_id}_sample_{sample_id}.pt')
                for sample_id in sample_ids]
            atom37_traj, model_traj, pred_positions, prmsd_final, prmsd, pred_trans_1, pred_rotmats_1, input_for_confidence = interpolant.sample(
                num_batch, 
                sample_length, 
                self.model,
                batch,
                save_all_repr=True,

            )
            def extract_i(all_input_for_confidence, i):
                out = {}

                for key, value in all_input_for_confidence.items():
                    if key == "input_for_confidence":
                        # 내부 딕셔너리 처리
                        sub_dict = {}
                        for subkey, subval in value.items():
                            sub_dict[subkey] = subval[i]
                        out[key] = sub_dict
                    else:
                        out[key] = value[i]

                return out
            
            for i in range(num_batch):
                sample_file = sample_files[i]
                input_for_confidence = extract_i(input_for_confidence, i)
                eu.save_conf_repr(input_for_confidence, sample_file)