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
from models.flow_model import FlowModel
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

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

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

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_chi_angle = noisy_batch['chi_angles_sin_cos']
        gt_atom14_pos = noisy_batch['atom14_gt_positions']
        alt_atom14_pos = noisy_batch['atom14_alt_gt_positions']
        gt_pseudo_beta = noisy_batch['pseudo_beta']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaN encountered in gt_rot_vf')

        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t']
        print(f'r3_t: {r3_t}')
        so3_t = noisy_batch['so3_t']
        print(f'so3_t: {so3_t}')
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        gt_atom14_pos *= training_cfg.bb_atom_scale / r3_norm_scale[..., None] # scaling 
        gt_bb_atoms = gt_atom14_pos[:, :, :3] # scaling 

        alt_atom14_pos *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        gt_pseudo_beta *= training_cfg.bb_atom_scale / r3_norm_scale

        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_angles_list = model_output['all_atom_preds']['angles']
        pred_unnormalized_angles_list = model_output['all_atom_preds']['unnormalized_angles']
        pred_atom_14_list = model_output['all_atom_preds']['positions']
        pred_rmsd = model_output['all_atom_preds']['prmsd']

        pred_atom_14_list = [pred * training_cfg.bb_atom_scale / r3_norm_scale[..., None] for pred in pred_atom_14_list]
        pred_atom_14_list = torch.stack(pred_atom_14_list, dim=0) # (O, B, L, A, 3)

        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')

        # Get the renamed ground truth 
        renamed_dict = compute_renamed_ground_truth(noisy_batch,
                                                    atom14_pred_positions=model_output['all_atom_preds']["positions"][-1])
        alt_naming_is_better = renamed_dict['alt_naming_is_better']
        renamed_atom14_gt_positions = renamed_dict['renamed_atom14_gt_positions']
        renamed_atom14_gt_exists = renamed_dict['renamed_atom14_gt_exists']

        renamed_atom14_gt_positions *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]

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

        # Backbone atom loss
        bb_atom_loss = compute_rmsd(pred_atom_14_list[:, :, :, :3],
                                renamed_atom14_gt_positions[:, :, :3],
                                cdr_mask=noisy_batch['diffuse_mask'],
                                atom14_gt_exists=noisy_batch['atom14_gt_exists'],
                                mode='bb',
                                compute_non_cdr=False
                                )
        # sc atom loss 
        sc_atom_loss = compute_rmsd(pred_atom_14_list[:, :, :, 3:],
                                renamed_atom14_gt_positions[:, :, 3:],
                                cdr_mask=noisy_batch['diffuse_mask'],
                                atom14_gt_exists=noisy_batch['atom14_gt_exists'],
                                mode='sc',
                                compute_non_cdr=True
                                )        

        # torsion angle loss 
        chi_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        if training_cfg.aux_loss_use_chi_loss:
            chi_loss = supervised_chi_loss(pred_angles_list,
                                        pred_unnormalized_angles_list,
                                        noisy_batch['aatype'],
                                        noisy_batch['res_mask'],
                                        noisy_batch['chi_mask'],
                                        gt_chi_angle,
                                        chi_weight=0.5,
                                        angle_norm_weight=0.02,
                                        cdr_mask=noisy_batch['diffuse_mask']
                                        )

        # final layer rmsd loss
        final_bb_rmsd = compute_rmsd(pred_atom_14_list[-1, :, :, :3].unsqueeze(0),
                                gt_atom14_pos[:, :, :3],
                                cdr_mask=noisy_batch['diffuse_mask'],
                                atom14_gt_exists=noisy_batch['atom14_gt_exists'],
                                mode='bb',
                                compute_non_cdr=False
                                )
        final_sc_rmsd = compute_rmsd(pred_atom_14_list[-1, :, :, 3:].unsqueeze(0),
                                gt_atom14_pos[:, :, 3:],
                                cdr_mask=noisy_batch['diffuse_mask'],
                                atom14_gt_exists=noisy_batch['atom14_gt_exists'],
                                mode='sc',
                                compute_non_cdr=True
                                )
        final_layer_rmsd = final_bb_rmsd * (training_cfg.aux_loss_bb_atom_loss_weight/2) + final_sc_rmsd * (training_cfg.aux_loss_sc_atom_loss_weight/2)

        # Pairwise distance loss
        dist_mat_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        if training_cfg.aux_loss_use_pair_loss:
            pred_atom_14 = pred_atom_14_list[-1]
            pred_bb_atoms = pred_atom_14[:, :, :3]
            gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
            gt_pair_dists = torch.linalg.norm(
                gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
            pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
            pred_pair_dists = torch.linalg.norm(
                pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

            flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
            flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
            flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
            flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

            gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
            pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
            pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

            dist_mat_loss = torch.sum(
                (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
                dim=(1, 2))
            dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        # local Pairwise distance loss
        local_dist_mat_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        if training_cfg.aux_loss_use_local_dist_mat_loss:
            pred_atom_14 = pred_atom_14_list[-1]
            local_dist_mat_loss = local_distance_loss(
                noisy_batch['aatype'],
                pred_atom_14, # scaled 
                renamed_dict['renamed_atom14_gt_exists'],
                renamed_atom14_gt_positions, # scaled 
                noisy_batch['diffuse_mask'][0],
                gt_pseudo_beta # scaled
            )
        
        # all atom clash loss 
        all_atom_clash_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        if training_cfg.aux_loss_use_all_atom_clash_loss:
            all_atom_clash_loss = compute_all_atom_clash_loss(
                                                            model_output['all_atom_preds']['positions'][-1],
                                                            noisy_batch['atom14_gt_exists'],
                                                            noisy_batch['res_idx'],
                                                            noisy_batch['residx_atom14_to_atom37'])
        
        # # backbone fape loss 
        bb_fape_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        if training_cfg.aux_loss_use_fape_bb_loss:
            bb_fape_loss = backbone_fape_loss(
                backbone_rigid_tensor=noisy_batch['backbone_rigid_tensor'],
                backbone_rigid_mask=noisy_batch['backbone_rigid_mask'],
                traj=model_output['all_atom_preds']['rigids'],
                cdr_mask=noisy_batch['diffuse_mask'],
                use_clamped_fape=False,
                # clamp_distance=10,
                # loss_unit_distance=10,
                # intercdr_distance=30
            )

        # # sidechain fape loss 
        sc_fape_loss = torch.zeros(gt_atom14_pos.shape[0], device=gt_atom14_pos.device)
        if training_cfg.aux_loss_use_fape_sc_loss:
            sc_fape_loss = sidechain_fape_loss(
                sidechain_frames=model_output['all_atom_preds']['sidechain_frames'],
                sidechain_atom_pos=model_output['all_atom_preds']['positions'],
                rigidgroups_gt_frames=noisy_batch['rigidgroups_gt_frames'],
                rigidgroups_alt_gt_frames=noisy_batch['rigidgroups_alt_gt_frames'],
                rigidgroups_gt_exists=noisy_batch['rigidgroups_gt_exists'],
                renamed_atom14_gt_positions=renamed_dict['renamed_atom14_gt_positions'],
                renamed_atom14_gt_exists=renamed_dict['renamed_atom14_gt_exists'],
                alt_naming_is_better=noisy_batch['alt_naming_is_better'],
                cdr_mask=noisy_batch['diffuse_mask'],
                use_clamped_fape=False
            )

        # calculate auxiliary loss 
        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss * training_cfg.aux_loss_bb_atom_loss_weight
            + sc_atom_loss * training_cfg.aux_loss_use_sc_atom_loss * training_cfg.aux_loss_sc_atom_loss_weight
            + chi_loss * training_cfg.aux_loss_use_chi_loss * training_cfg.aux_loss_chi_loss_weight 
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss * training_cfg.aux_loss_pair_loss_weight
            + local_dist_mat_loss * training_cfg.aux_loss_use_local_dist_mat_loss * training_cfg.aux_loss_local_dist_mat_loss_weight
            + final_layer_rmsd * training_cfg.aux_loss_use_final_layer_rmsd * training_cfg.aux_loss_final_layer_rmsd_weight
            + bb_fape_loss * training_cfg.aux_loss_use_fape_bb_loss * training_cfg.aux_loss_fape_bb_loss_weight 
            + sc_fape_loss * training_cfg.aux_loss_use_fape_sc_loss * training_cfg.aux_loss_fape_sc_loss_weight 
        )

        # calculate prmsd 
        if self._model_cfg.prmsd.use_prmsd:
            prmsd_loss = compute_prmsd_loss(pdev=pred_rmsd,
                                    pred_position=pred_atom_14, # predicted structure (b, l, 14, 3)
                                    atom14_gt_positions=gt_atom14_pos, # gt stucture  (b, l, 14, 3)
                                    cdr_mask=noisy_batch['diffuse_mask']) # (b, l)


            auxiliary_loss += prmsd_loss * training_cfg.aux_loss_use_prmsd_loss

        auxiliary_loss *= (
            (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=10)

        se3_vf_loss += auxiliary_loss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN loss encountered')

        # print({
        #     "trans_loss": trans_loss,
        #     "auxiliary_loss": auxiliary_loss,
        #     "rots_vf_loss": rots_vf_loss,
        #     "se3_vf_loss": se3_vf_loss,
        #     "bb_atom_loss": bb_atom_loss,
        #     'sc_atom_loss': sc_atom_loss,
        #     'chi_loss': chi_loss,
        #     'dist_mat_loss': dist_mat_loss,
        #     'all_atom_clash_loss': all_atom_clash_loss,
        #     'local_dist_mat_loss': local_dist_mat_loss,
        #     'bb_fape_loss': bb_fape_loss,
        #     'sc_fape_loss': sc_fape_loss
        # })
        return {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "bb_atom_loss": bb_atom_loss,
            'sc_atom_loss': sc_atom_loss,
            'chi_loss': chi_loss,
            'dist_mat_loss': dist_mat_loss,
            'all_atom_clash_loss': all_atom_clash_loss,
            'local_dist_mat_loss': local_dist_mat_loss,
            'bb_fape_loss': bb_fape_loss,
            'sc_fape_loss': sc_fape_loss
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch['diffuse_mask']
        csv_idx = batch['csv_idx']
        raw_path = batch['raw_path']
        pdb_id = raw_path.split('/')[-1].replace('.pdb', '')
        atom37_traj, clean_atom37_traj, pred_positions, prmsd, pred_trans_1, pred_rotmats_1 = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            aatype=batch['aatype'],
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            diffuse_mask=diffuse_mask,
            chain_idx=batch['chain_idx'],
            res_idx=batch['res_idx'],
            pair_init=batch['pair_init']
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
            saved_path = au.write_prot_to_pdb(
                final_pos,
                file_path=os.path.join(sample_dir, pdb_id+'.pbd'),
                aatype=batch['aatype'].cpu(),
                chain_index=batch['chain_idx'].cpu(),
                no_indexing=False,
                overwrite=True
            )
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
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # 사용되지 않은 파라미터 찾기
        params_after = {n: p.requires_grad for n, p in self.named_parameters() if p.requires_grad}
        unused_params = [n for n in params_before if not params_after[n]]
        
        if unused_params:
            print(f"⚠️ Unused parameters: {unused_params}")
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(), self._exp_cfg.optimizer.max_lr
        )
        scheduler = self.get_cosine_scheduler_w_warmup(
            optimizer,
            self._exp_cfg.optimizer.warmup_steps,
            self._exp_cfg.optimizer.decay_steps,
            self._exp_cfg.optimizer.min_lr,
            self._exp_cfg.optimizer.max_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def predict_step(self, batch, batch_idx):
        del batch_idx # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)

        sample_ids = batch['sample_id'].squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)


        pdb_id = batch['raw_path'].split('/')[-1].replace('.pdb', '')

        if 'diffuse_mask' in batch: # motif-scaffolding
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            diffuse_mask = batch['diffuse_mask']

            true_bb_pos = all_atom.atom37_from_trans_rot(trans_1, rotmats_1, 1 - diffuse_mask)
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            _, sample_length, _ = trans_1.shape
            sample_dirs = [os.path.join(
                self.inference_dir, pdb_id, f'sample_{sample_id}')
                for sample_id in sample_ids]
        else: # unconditional
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            sample_dirs = [os.path.join(
                self.inference_dir, f'length_{sample_length}', f'{pdb_id}')
                for sample_id in sample_ids]
            trans_1 = rotmats_1 = diffuse_mask = None
            diffuse_mask = torch.ones(1, sample_length, device=device)

        # Sample batch
        atom37_traj, model_traj, pred_positions, prmsd, pred_trans_1, pred_rotmats_1 = interpolant.sample(
            num_batch, sample_length, self.model,
            aatype=batch['aatype'],
            trans_1=trans_1, rotmats_1=rotmats_1, diffuse_mask=diffuse_mask,
            pair_init=batch['pair_init']
        )
        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        pred_positions_37 = []
        pred_positions = du.to_numpy(pred_positions)
        for i in range(pred_positions.shape[0]):
            pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch)
            pred_positions_37.append(pred_position_37)
        
        pred_positions = np.stack(pred_positions_37)
        prmsds = du.to_numpy(prmsd)

        for i in range(num_batch):
            sample_dir = sample_dirs[i]
            pred_position = pred_positions[i]
            prmsd = prmsds[i]
            bb_traj = bb_trajs[i]
            os.makedirs(sample_dir, exist_ok=True)
            if 'aatype' in batch:
                aatype = du.to_numpy(batch['aatype'].long())[0]
            else:
                aatype = np.zeros(sample_length, dtype=int)

            if 'chain_idx' in batch:
                chain_idx = du.to_numpy(batch['chain_idx'].long())[0]
            else:
                chain_idx = None
            _ = eu.save_traj(
                sample=pred_position,
                bb_prot_traj=bb_traj,
                x0_traj=np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                b_factors=prmsd,
                output_dir=sample_dir,
                aatype=aatype,
                chain_index=chain_idx,
                save_traj_bool=self._interpolant_cfg.save_traj
            )
