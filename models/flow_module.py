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
from models.confidence import ConfidenceHead
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom
from data.motif_index import find_anchor
from experiments import utils as eu
from models.loss import *
from openfold.utils.loss import between_residue_bond_loss
import sys 

sys.stdout.flush()

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
        
        if self._model_cfg.train_confidence:
            self._confidence_cfg = cfg.model.confidence_head
            self.confidence_model = ConfidenceHead(
                n_blocks=self._confidence_cfg.n_blocks,
                c_z=self._confidence_cfg.c_z,
                c_s_inputs=self._confidence_cfg.c_s_inputs,
                b_pae=self._confidence_cfg.b_pae,
                b_plddt=self._confidence_cfg.b_plddt,
                max_atoms_per_token=self._confidence_cfg.max_atoms_per_token,
                pairformer_dropout=self._confidence_cfg.pairformer_dropout,
                blocks_per_ckpt=self._confidence_cfg.blocks_per_ckpt,
                distance_bin_start=self._confidence_cfg.distance_bin_start,
                distance_bin_end=self._confidence_cfg.distance_bin_end,
                distance_bin_step=self._confidence_cfg.distance_bin_step,
                stop_gradient=self._confidence_cfg.stop_gradient,
            )
            self.rollout = Interpolant(cfg.rollout)
            self._plddt_cfg = cfg.experiment.training.plddt
            self.plddt_loss_module = PLDDTLoss(
                min_bin=self._plddt_cfg.min_bin,
                max_bin=self._plddt_cfg.max_bin,
                no_bins=self._plddt_cfg.no_bins,
                is_not_nucleotide_threshold=self._plddt_cfg.threshold,
                eps=self._plddt_cfg.eps,
                normalize=self._plddt_cfg.normalize,
                reduction=self._plddt_cfg.reduction,
            )
            self._pae_cfg = cfg.experiment.training.pae
            self.pae_loss_module = PAELoss(
                min_bin=self._pae_cfg.min_bin,
                max_bin=self._pae_cfg.max_bin,
                no_bins=self._pae_cfg.no_bins,
                eps=self._pae_cfg.eps,
                reduction=self._pae_cfg.reduction,
            )

        # Set-up interpolant
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

    def model_step(self, noisy_batch, N_cycle):
        training_cfg = self._exp_cfg.training
        loss_loop_mask = noisy_batch['res_mask'] * noisy_batch['loop_mask']
        loss_diffuse_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        loss_atom_diffuse_mask = noisy_batch['atom_diffuse_mask']
    
        if torch.any(torch.sum(loss_loop_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_loop_mask.shape
        device = noisy_batch['trans_1'].device 

        # Ground truth labels
        r_1 = noisy_batch['r_1']
        gt_atom14_pos = noisy_batch['atom14_gt_positions'].clone()
        alt_atom14_pos = noisy_batch['atom14_alt_gt_positions'].clone()
        gt_pseudo_beta = noisy_batch['pseudo_beta'].clone()

        print("raw_path", noisy_batch['raw_path'])

        # Timestep used for normalization.
        t = noisy_batch['t']
        r3_norm_scale = 1 - torch.min(
            t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        gt_atom14_pos = gt_atom14_pos * training_cfg.atom_scale / r3_norm_scale[..., None] # scaled
        gt_bb_atoms = gt_atom14_pos[:, :, :3] # scaled
        alt_atom14_pos = alt_atom14_pos * training_cfg.atom_scale / r3_norm_scale[..., None]
        gt_pseudo_beta = gt_pseudo_beta * training_cfg.atom_scale / r3_norm_scale


        ### calculate loss ###
        if not hasattr(self, "confidence_model"):
            # Model output predictions.
            model_output = self.model(noisy_batch, N_cycle)
            pred_r_1 = model_output['pred_r_1'].clone() 
            pred_atom_14 = model_output['pred_r_1_unflatten'].clone() # scaled 
            pred_distogram = model_output['pair_outputs'].clone()

            pred_atom_14 = pred_atom_14 * training_cfg.atom_scale / r3_norm_scale[..., None]
            pred_bb_atoms = pred_atom_14[:, :, :3]
            
            # Get the renamed ground truth 
            renamed_dict = compute_renamed_ground_truth(
                noisy_batch,
                atom14_pred_positions=model_output['pred_r_1_unflatten']
                )

            renamed_atom14_gt_exists = renamed_dict['renamed_atom14_gt_exists'].clone()
            renamed_atom14_gt_positions = renamed_dict['renamed_atom14_gt_positions'] * training_cfg.atom_scale / r3_norm_scale[..., None]

            # Translation VF loss
            r3_error = (r_1 - pred_r_1) / r3_norm_scale * training_cfg.r3_scale

            loss_atom_denom = torch.sum(loss_atom_diffuse_mask, dim=-1) * 3
            r3_loss = training_cfg.r3_loss_weight * torch.sum(
                torch.abs(r3_error) * loss_atom_diffuse_mask[..., None],
                dim=(-1, -2)
            ) / loss_atom_denom
            r3_loss = torch.clamp(r3_loss, max=40)

            # calculate pair feature loss (beta carbon contact prob)
            distogram_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
            if training_cfg.use_local_pair_feat_loss and self._model_cfg.distogram_head.use_pair_head:
                distogram_loss = calc_distogram_loss(
                    pred_distogram=pred_distogram, # non-scaled 
                    gt_pos=noisy_batch['trans_1'],
                    res_mask=noisy_batch['res_mask'],
                    neighbor_indices=None,
                    cdr_residues=None,
                    min_bin=self._model_cfg.distogram_head.min_bin,
                    max_bin=self._model_cfg.distogram_head.max_bin,
                    num_bins=self._model_cfg.distogram_head.num_bins
                )

            # distance map loss
            dist_mat_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
            if training_cfg.use_dist_mat_loss:
                gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
                pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])

                gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1) # (B, L*3, L*3)
                gt_pair_dists = torch.clamp(gt_pair_dists, max=22)
                pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1) # (B, L*3, L*3)
                pred_pair_dists = torch.clamp(pred_pair_dists, max=22)

                # diffuse_mask를 pair mask로 확장
                flat_diffuse_mask = loss_diffuse_mask[:, :, None].repeat(1, 1, 3)  # (B, L, 3)
                flat_diffuse_mask = flat_diffuse_mask.reshape(num_batch, num_res*3)  # (B, L*3)
                pair_dist_mask = (flat_diffuse_mask[:, :, None] + flat_diffuse_mask[:, None, :]) > 0 # (B, L*3, L*3)
                pair_dist_mask = pair_dist_mask.float()

                # loss 계산
                dist_mat_loss = torch.sum(torch.abs(gt_pair_dists - pred_pair_dists) * pair_dist_mask, dim=(1,2))
                dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1,2)) + 1)

            # Loop Backbone RMSD loss
            bb_loop_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
            if training_cfg.use_loop_bb_loss:
                bb_loop_loss = compute_rmsd(
                    pred_atom_14[None, ...],
                    renamed_atom14_gt_positions,
                    mask=noisy_batch['loop_mask'],
                    atom14_gt_exists=renamed_atom14_gt_exists,
                    mode='bb',
                    data_mode=noisy_batch['mode'],
                    compute_unmasked=False,
                    mask_clamp=30
                    )     

            # Loop Sidechain RMSD loss
            sc_loop_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
            if training_cfg.use_loop_sc_loss:
                sc_loop_loss = compute_rmsd(
                    pred_atom_14[None, ...],
                    renamed_atom14_gt_positions,
                    mask=noisy_batch['loop_mask'],
                    atom14_gt_exists=renamed_atom14_gt_exists,
                    mode='sc',
                    data_mode=noisy_batch['mode'],
                    compute_unmasked=False,
                    mask_clamp=30
                    ) 

            # local Pairwise distance loss (final layer만 계산)
            local_dist_mat_loss = torch.zeros(gt_atom14_pos.shape[0], device=device)
            if training_cfg.use_local_dist_mat_loss and noisy_batch['mode'] not in ["monomer", "polymer"]:
                local_dist_mat_loss, neighbor_indices, cdr_residues = local_distance_loss(
                    pred_atom_14, # scaled 
                    renamed_atom14_gt_exists,
                    renamed_atom14_gt_positions, # scaled
                    noisy_batch['cdr_residues'],
                    noisy_batch['neighbor_indices']
                )   # local_loss_mask: (B, N, N, 14)
                
            # all atom clash loss 
            batch_size = gt_atom14_pos.shape[0]
            all_atom_clash_loss = torch.zeros(batch_size, device=device)
            if training_cfg.use_all_atom_clash_loss:
                all_atom_clash_loss = compute_all_atom_clash_loss(
                    model_output['pred_r_1_unflatten'],
                    noisy_batch['atom14_gt_exists'],
                    noisy_batch['residue_index'],
                    noisy_batch['residx_atom14_to_atom37'],
                    interface_mask=None,
                )


            within_clash_loss = torch.zeros(batch_size, device=device)
            if training_cfg.use_within_clash_loss:
                within_clash_loss = compute_within_clash_loss(
                    model_output['pred_r_1_unflatten'],
                    noisy_batch['atom14_gt_exists'],
                    noisy_batch['loop_mask'],
                    noisy_batch['aatype']
                    )   

            # bond loss 
            bond_length_loss = torch.zeros(batch_size, device=device)
            ca_c_n_loss = torch.zeros(batch_size, device=device)
            c_n_ca_loss = torch.zeros(batch_size, device=device)
            if training_cfg.use_bond_loss:
                bond_loss_info = between_residue_bond_loss(
                    pred_atom_positions=model_output['pred_r_1_unflatten'],
                    pred_atom_mask=noisy_batch['atom14_gt_exists'],
                    residue_index=noisy_batch['residue_index'],
                    aatype=noisy_batch['aatype']
                )
                bond_length_loss = bond_loss_info['c_n_loss_mean']
                ca_c_n_loss = bond_loss_info['ca_c_n_loss_mean']
                c_n_ca_loss = bond_loss_info['c_n_ca_loss_mean']

            # calculate auxiliary loss 
            total_loss = r3_loss + distogram_loss * training_cfg.distogram_loss_weight

            bb_auxiliary_loss = (
                dist_mat_loss * training_cfg.dist_mat_loss_weight
                + bb_loop_loss * training_cfg.loop_bb_loss_weight
            )
            sc_auxiliary_loss = (
                sc_loop_loss * training_cfg.loop_sc_loss_weight
                + local_dist_mat_loss * training_cfg.local_dist_mat_loss_weight
            )

            # calculate violation loss
            bb_violation_loss = (
                + bond_length_loss * training_cfg.bond_length_loss_weight
                + ca_c_n_loss * training_cfg.ca_c_n_loss_weight 
                + c_n_ca_loss * training_cfg.c_n_ca_loss_weight
            )
            sc_violation_loss = (
                all_atom_clash_loss * training_cfg.all_atom_clash_loss_weight 
                + within_clash_loss * training_cfg.within_clash_loss_weight
            )

            bb_auxiliary_loss *= (
                (t[:, 0] > training_cfg.bb_aux_loss_t_pass)
            )
            sc_auxiliary_loss *= (
                (t[:, 0] > training_cfg.sc_aux_loss_t_pass)
            )
            auxiliary_loss = bb_auxiliary_loss + sc_auxiliary_loss
            auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
            auxiliary_loss = torch.clamp(auxiliary_loss, max=30)

            bb_violation_loss *= (
                (t[:, 0] > training_cfg.bb_viol_loss_t_pass)
            )
            sc_violation_loss *= (
                (t[:, 0] > training_cfg.sc_viol_loss_t_pass)
            )
            violation_loss = bb_violation_loss + sc_violation_loss
            violation_loss *= self._exp_cfg.training.viol_loss_weight 
            violation_loss = torch.clamp(violation_loss, max=10)

            total_loss = total_loss + auxiliary_loss + violation_loss

            return {
                "total_loss": total_loss,
                "r3_loss": r3_loss,
                "distogram_loss": distogram_loss,
                "auxiliary_loss": auxiliary_loss,
                'dist_mat_loss': dist_mat_loss,
                "bb_loop_loss": bb_loop_loss,
                'sc_loop_loss': sc_loop_loss,
                'local_dist_mat_loss': local_dist_mat_loss,
                'all_atom_clash_loss': all_atom_clash_loss,
                'within_clash_loss': within_clash_loss,
                'bond_length_loss': bond_length_loss,
                'ca_c_n_loss': ca_c_n_loss,
                'c_n_ca_loss': c_n_ca_loss,
                'violation_loss': violation_loss
            }
        
        else:
            self.rollout.set_device(loss_loop_mask.device)
            with torch.no_grad():
                s_init, z_init = self.model.embed_input(noisy_batch)
                s_init, s, z, pair_outputs = self.model.do_pairformer(
                    s_init, 
                    z_init, 
                    noisy_batch['edge_mask'][0][None, ...], 
                    self._model_cfg.pairformer.n_cycles, 
                    noisy_batch['edge_mask'].shape[0]
                    )
                r3_traj, clean_atom37_traj, pred_positions, pred_trans_1 = self.rollout.sample(
                    self.model,
                    noisy_batch,
                    s_init,
                    s,
                    z
                )    

            b, N_atom, _ = noisy_batch['r_1'].shape    
            _, N_res, _, = noisy_batch['atom14_gt_exists'].shape
            plddt_pred, pae_pred = self.confidence_model(
                noisy_batch['ref_feature_dict'],
                s_init[0],
                s[0],
                z[0],
                noisy_batch['edge_mask'][0],
                pred_trans_1,
                )       
            
            # make masks for confidence loss
            # # for plddt  
            coordinate_mask = du.atom_flatten(
                noisy_batch['atom14_gt_exists'], 
                noisy_batch['atom14_gt_exists']
                )[0] # (N_atom)
            rep_atom_mask = du.atom_flatten(
                torch.eye(14, device=plddt_pred.device)[1].repeat(num_res, 1)[None, ...],
                noisy_batch['atom14_gt_exists'][0][None, ...]
                )[0] # (N_atom)
            
            # # for pae 
            has_frame = noisy_batch['atom14_gt_exists'][..., 0:3].all(dim=-1)[0].to(torch.int) # (L_res)
            atom_flat = noisy_batch['atom14_gt_exists'].view(b, -1) # (B, L_res * 14)
            atom_global_index = torch.cumsum(atom_flat, dim=-1) - 1 # (B, L_res * 14)
            atom_global_index = atom_global_index * atom_flat # (B, L_res * 14)
            atom_global_index = torch.where(atom_flat.bool(), atom_global_index, torch.zeros_like(atom_global_index)) # (B, L_res * 14)
            backbone_local = torch.tensor([0,1,2], device=noisy_batch['atom14_gt_exists'].device)
            atom_global_index = atom_global_index.view(num_batch, num_res, 14)
            frame_atom_index = (atom_global_index[0, :, backbone_local] * has_frame.unsqueeze(-1)).to(torch.long) # (L_res, 3)

            diffuse_mask_i = noisy_batch["diffuse_mask"][:, :, None]  # (B, L_res, 1)
            diffuse_mask_j = noisy_batch["diffuse_mask"][:, None, :]  # (B, 1, L_res)
            inter_mask = 1 - (diffuse_mask_i == diffuse_mask_j).float() # 0: intra / 1: inter (B, L_res, L_res)

            # calculate confidence loss 
            plddt_loss = self.plddt_loss_module(
                logits=plddt_pred,
                pred_coordinate=r3_traj[-1],
                true_coordinate=noisy_batch['r_1'][0],
                coordinate_mask=coordinate_mask,
                is_nucleotide=torch.zeros(N_atom, device=plddt_pred.device),
                is_polymer=torch.ones(N_atom, device=plddt_pred.device),
                rep_atom_mask=rep_atom_mask,
                atom_diffuse_mask=noisy_batch['atom_diffuse_mask']
            )

            pae_loss = self.pae_loss_module(
                logits=pae_pred,
                pred_coordinate=r3_traj[-1],
                true_coordinate=noisy_batch['r_1'][0],
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
                frame_atom_index=frame_atom_index,
                has_frame=has_frame,
                inter_mask=inter_mask
            )
            
            total_loss = plddt_loss + pae_loss
            return {
                "total_loss": total_loss,
                "plddt_loss": plddt_loss,
                "pae_loss": pae_loss
            }
        
    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        b = res_mask.shape[0]
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        num_batch, num_atom, _ = batch['r_1'].shape
        edge_mask = batch['edge_mask']
        loop_mask = batch['loop_mask']
        diffuse_mask = batch['diffuse_mask']
        raw_path = batch['raw_path']
        pdb_id = raw_path.split('/')[-1].replace('.pdb', '')

        s_init, z_init = self.model.embed_input(batch)
        s_init, s, z, pair_outputs = self.model.do_pairformer(s_init, z_init, edge_mask[0][None, ...], self._model_cfg.pairformer.n_cycles, b)
        r3_traj, clean_atom37_traj, pred_positions, pred_trans_1 = self.interpolant.sample(
            self.model,
            batch,
            s_init,
            s,
            z
        )

        if hasattr(self, "confidence_model"):
            plddt_pred, pae_pred = self.confidence_model(
                batch['ref_feature_dict'],
                s_init[0],
                s[0],
                z[0],
                batch['edge_mask'][0],
                pred_trans_1,
            )       
        pred_positions_37 = []
        pred_positions = du.to_numpy(pred_positions)
        for i in range(b):
            pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch)
            pred_positions_37.append(pred_position_37)
        
        pred_positions = np.stack(pred_positions_37)
        batch_metrics = []
        
        sample_dir = os.path.join(
            self.checkpoint_dir,
            f'{pdb_id}_len_{num_res}'
        )
        os.makedirs(sample_dir, exist_ok=True)

        if not hasattr(self, "confidence_model"):
            b_factor_alt = diffuse_mask.cpu().numpy()
            b_factors = np.tile((b_factor_alt * 100)[:, :, None], (1, 1, 37)) # (B, L, 37)
        else:
            plddt_bins = plddt_pred.shape[-1]
            plddt_probs = nn.functional.softmax(plddt_pred, dim=-1)  # [B, N_atom, plddt_bins]
            bin_values = torch.linspace(0, 1, plddt_bins, device=plddt_pred.device)  # [plddt_bins]
            plddt_score = torch.sum(plddt_probs * bin_values, dim=-1)  # [B, N_atom]
            b_factors_14 = du.atom_unflatten(plddt_score, batch['atom14_gt_exists']) # [B, N_token, 14]
            b_factors_14 = du.to_numpy(b_factors_14) * 100

            b_factors = []
            for i in range(b):
                b_factors_37 = all_atom.atom14_to_atom37(b_factors_14[i][..., None], batch)
                b_factors.append(np.squeeze(b_factors_37, axis=-1))
            b_factors = np.stack(b_factors)

        for i in range(num_batch):
            # Write out sample to PDB file (wo b-factors)
            final_pos = pred_positions[i]
            b_factor = b_factors[i]
            
            if batch['mode'] == 'polymer' or batch['mode'] == 'monomer':
                unique_vals, mapped = torch.unique(batch['chain_idx'][0], return_inverse=True)
                batch['chain_idx'] = mapped.unsqueeze(0).expand(b, -1)

            saved_path = au.write_prot_to_pdb(
                final_pos,
                file_path=os.path.join(sample_dir, pdb_id+'.pbd'),
                aatype=batch['aatype'].cpu(),
                chain_index=batch['chain_idx'].cpu(),
                no_indexing=False,
                overwrite=True,
                b_factors=b_factor
            )
        
        if not hasattr(self, "confidence_model"):
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
            b, N = loop_mask.shape
            h3_mask = torch.zeros_like(loop_mask)
            count = 0
            for i in range(b):
                count = 0  
                in_group = False  
                group_start = None  
                
                # 연속된 1들의 그룹을 추적
                for j in range(N):
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

        else:
            # make masks for confidence loss 
            # # for plddt 
            coordinate_mask = du.atom_flatten(
                batch['atom14_gt_exists'], 
                batch['atom14_gt_exists']
                )[0] # (N_atom)
            rep_atom_mask = du.atom_flatten(
                torch.eye(14, device=plddt_pred.device)[1].repeat(num_res, 1)[None, ...],
                batch['atom14_gt_exists'][0][None, ...]
                )[0] # (N_atom)
            
            # # for pae
            has_frame = batch['atom14_gt_exists'][..., 0:3].all(dim=-1)[0].to(torch.int) # (L_res)
            atom_flat = batch['atom14_gt_exists'].view(b, -1) # (B, L_res * 14)
            atom_global_index = torch.cumsum(atom_flat, dim=-1) - 1 # (B, L_res * 14)
            atom_global_index = atom_global_index * atom_flat # (B, L_res * 14)
            atom_global_index = torch.where(atom_flat.bool(), atom_global_index, torch.zeros_like(atom_global_index)) # (B, L_res, 14)
            backbone_local = torch.tensor([0,1,2], device=batch['atom14_gt_exists'].device)
            atom_global_index = atom_global_index.view(num_batch, num_res, 14)
            frame_atom_index = (atom_global_index[0, :, backbone_local] * has_frame.unsqueeze(-1)).to(torch.long) # (L_res, 3)

            diffuse_mask_i = batch["diffuse_mask"][:, :, None]  # (B, L_res, 1)
            diffuse_mask_j = batch["diffuse_mask"][:, None, :]  # (B, 1, L_res)
            inter_mask = 1 - (diffuse_mask_i == diffuse_mask_j).float() # 0: intra / 1: inter

            # calculate loss 
            plddt_loss = self.plddt_loss_module(
                logits=plddt_pred,
                pred_coordinate=r3_traj[-1],
                true_coordinate=batch['r_1'][0],
                coordinate_mask=coordinate_mask,
                is_nucleotide=torch.zeros(num_atom, device=plddt_pred.device),
                is_polymer=torch.ones(num_atom, device=plddt_pred.device),
                rep_atom_mask=rep_atom_mask,
                atom_diffuse_mask=batch['atom_diffuse_mask']
            )
            batch_metrics.append({"plddt_loss": plddt_loss})
            pae_loss = self.pae_loss_module(
                logits=pae_pred,
                pred_coordinate=r3_traj[-1],
                true_coordinate=batch['r_1'][0],
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
                frame_atom_index=frame_atom_index,
                has_frame=has_frame,
                inter_mask=inter_mask
            )
            batch_metrics.append({"pae_loss": pae_loss})
            
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
    
    def training_step(self, batch: Any, stage: int):
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        N_cycle = random.randint(1, self._model_cfg.pairformer.n_cycles)
        batch_losses = self.model_step(noisy_batch, N_cycle)

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

        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        for loss_name, loss_dict in batch_losses.items():
            batch_t = t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
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
        if not hasattr(self, "confidence_model"):
            parameters = self.model.parameters()
        else:
            parameters = self.confidence_model.parameters()
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
        return {"optimizer": optimizer, 
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
        pdb_id = batch['processed_path'].split('/')[-1].replace('.pdb', '')

        sample_root_dir = os.path.join(self.inference_dir, pdb_id)
        if not os.path.exists(sample_root_dir):
            os.makedirs(sample_root_dir, exist_ok=True)

        if pdb_id != self.current_pdb_id:
            self.pairformer_cache.clear()
            self.current_pdb_id = pdb_id

        if pdb_id in self.pairformer_cache:
            print(f"{pdb_id} pairformer cache will be used")
            s_init, s, z = self.pairformer_cache[pdb_id]
        else:
            print(f"new pdb {pdb_id}: make pairformer output")
            s_init_embed, z_init = self.model.embed_input(batch)
            s_init, s, z, pair_outputs = self.model.do_pairformer(
                s_init_embed,
                z_init,
                batch['edge_mask'][0][None, ...],
                self._model_cfg.pairformer.n_cycles,
                num_batch
            )
            # 결과를 캐시에 저장합니다.
            self.pairformer_cache[pdb_id] = (s_init, s, z)
            
            # save distogram 
            # find cdr_residues to mark cdr on the distogram 
            cdr_residues = find_anchor(batch["loop_mask"][0], only_h3=False)
            cdr_residues = [(cdr_residues[2*i], cdr_residues[2*i+1]) for i in range(6)]

            # pairformer distogram 시각화
            dist_map_pairformer = eu.dist_map_from_distogram(
                pair_outputs[0][None, ...],
                min_bin=self._model_cfg.distogram_head.min_bin,
                max_bin=self._model_cfg.distogram_head.max_bin,
                do_softmax=False
                ) # (1, N, N)
            eu.visualize_dist_map(
                dist_map_pairformer[0], 
                os.path.join(sample_root_dir, "pairformer_dist_ca.png"), 
                title=f"{pdb_id.upper()} Pairformer Cα Distogram",
                cdr_residues=cdr_residues,
                mark_cdr=True
                )

            # gt distogram 시각화 
            gt_ca_distogram = calc_distogram(  
                batch['trans_1'][0][None, ...],
                min_bin=self._model_cfg.distogram_head.min_bin,
                max_bin=self._model_cfg.distogram_head.max_bin,
                num_bins=self._model_cfg.distogram_head.num_bins,
            ) # (1, L, L, num_bins)
            gt_ca_dist = eu.dist_map_from_distogram(
                gt_ca_distogram, 
                min_bin=self._model_cfg.distogram_head.min_bin,
                max_bin=self._model_cfg.distogram_head.max_bin,
                do_softmax=False)
            eu.visualize_dist_map(
                gt_ca_dist[0], 
                os.path.join(sample_root_dir, "gt_dist_ca.png"), 
                title=f"{pdb_id.upper()} True Cα Distogram",
                cdr_residues=cdr_residues,
                mark_cdr=True
                )
            # gt distogram과 pairformer distogram 차이 
            diff_distance_map = np.abs(gt_ca_dist - dist_map_pairformer)

            eu.visualize_dist_map(
                diff_distance_map[0], 
                os.path.join(sample_root_dir, "diff_dist_ca.png"), 
                title=f"{pdb_id.upper()} |True - Pairformer|",
                cdr_residues=cdr_residues,
                mark_cdr=True,
                cmap='hot',
                vmax=22
                )

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
        for i in range(pred_positions.shape[0]):
            pred_position_37 = all_atom.atom14_to_atom37(pred_positions[i], batch) # (L_crop, 37, 3)
            pred_positions_37.append(pred_position_37)
        pred_positions = np.stack(pred_positions_37) # (B, L, 37, 3)

        samples = os.listdir(sample_root_dir)
        sample_nums = samples 
        next_sample_num = -1
        # protein의 n번째 (n>1) 배치를 생성할 때 
        if any('sample' in filename for filename in samples):
            sample_nums = sorted([int(sample.replace("sample_", "")) for sample in samples if 'sample' in sample])
            next_sample_num = sample_nums[-1]

        for i in range(num_batch):
            next_sample_num += 1
            sample_dir = os.path.join(sample_root_dir, f"sample_{next_sample_num}")
            pred_position = pred_positions[i]
            bb_traj = bb_trajs[i]
            os.makedirs(sample_dir, exist_ok=True)

            # save structure data 
            aatype = du.to_numpy(batch['aatype'][i].int())
            chain_idx = du.to_numpy(batch['chain_idx'][i].int())
            diffuse_mask = du.to_numpy(batch['diffuse_mask'][i].int())

            _ = eu.save_traj(
                sample=pred_position, # (L, 37, 3)
                bb_prot_traj=bb_traj, 
                x0_traj=np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                b_factors=None,  # 위의 prmsd 집어넣기 
                diffuse_mask=diffuse_mask,
                output_dir=sample_dir,
                aatype=aatype,
                chain_index=chain_idx,
                save_traj_bool=self._interpolant_cfg.save_traj
            )
            
