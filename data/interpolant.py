from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from torch import autograd
from motif_scaffolding import twisting
from models.loss import compute_prmsd, compute_all_atom_clash_loss, compute_within_clash_loss
import analysis.utils as au 

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def manage_missing_batch(self, xyz, mask):
        """for missing residues, get the closest residue coordinate (batch version)
        
        Args:
            xyz (tensor): xyz coordinates [b, L, 3]
            mask (tensor): mask information [b, L] (True = atom exists)

        Returns:
            xyz (tensor): modified xyz coordinate [b, L, 3]
        """
        b, L, _ = xyz.shape  # 배치 차원 고려

        # 존재하는 원소의 인덱스 찾기 (배치별)
        exist_in_xyz = [torch.where(mask[i])[0] for i in range(b)]  # 각 배치에서 존재하는 residue 인덱스 리스트

        # 새로운 xyz를 저장할 tensor
        new_xyz = xyz.clone()

        for i in range(b):
            if len(exist_in_xyz[i]) == 0:  # 모든 residue가 missing이면 건너뛰기
                continue

            # 존재하는 원소의 인덱스 리스트 (L_sub)
            valid_idx = exist_in_xyz[i]

            # 각 residue가 가장 가까운 존재하는 residue를 찾기 위한 거리 계산
            seqmap = (torch.arange(L, device=xyz.device)[:, None] - valid_idx[None, :]).abs()  # (L, L_sub)
            closest_idx = torch.argmin(seqmap, dim=-1)  # L -> 가장 가까운 residue의 valid_idx 내부 인덱스

            # 실제 index 가져오기 (L 크기의 인덱스 배열)
            nearest_residue_idx = valid_idx[closest_idx]

            # 가장 가까운 residue의 좌표 가져오기
            nearest_residue_coords = xyz[i, nearest_residue_idx]

            # mask가 False인 부분을 최근접 residue 좌표로 대체
            new_xyz[i] = torch.where(mask[i].unsqueeze(-1), xyz[i], nearest_residue_coords)

        return new_xyz
        
    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_0 = _centered_gaussian(*res_mask.shape, self._device)
        masked_trans = self.manage_missing_batch(trans_1, mask=~diffuse_mask.bool())
        trans_0 = trans_0 * du.NM_TO_ANG_SCALE

        trans_0 = trans_0 + masked_trans
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom
        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        so3_t = t
        r3_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['pair_init'] = batch['pair_init']

        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def sample(
            self,
            num_batch,
            num_res,
            model,
            batch,
            num_timesteps=None,
            trans_potential=None,
            trans_0=None,
            rotmats_0=None,
            verbose=False,
            save_all_repr=False,
            rollout=False
        ):

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                    num_batch, num_res, self._device)

            trans_0 *= du.NM_TO_ANG_SCALE

            masked_trans = self.manage_missing_batch(batch["trans_1"], mask=~batch["diffuse_mask"].bool())
            trans_0 = trans_0 + masked_trans
            trans_0 = _trans_diffuse_mask(trans_0, batch["trans_1"], batch["diffuse_mask"])

        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        motif_scaffolding = False
        diffuse_mask = batch['diffuse_mask']
        trans_1 = batch['trans_1']
        rotmats_1 = batch['rotmats_1']

        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
        if motif_scaffolding and not self._cfg.twisting.use: # amortisation
            diffuse_mask = diffuse_mask.expand(num_batch, -1) # shape = (B, num_residue)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError('NaN detected in trans_0')

        logs_traj = defaultdict(list)
        if motif_scaffolding and self._cfg.twisting.use: # sampling / guidance
            assert trans_1.shape[0] == 1 # assume only one motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = twisting.find_ranges_and_lengths(motif_locations)

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = true_motif_locations if self._cfg.twisting.motif_loc else None
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(num_res, motif_segments_length, motif_locations=motif_locations, num_rots=self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, trans_motif=trans_motif, R_motif=R_motif, max_offsets=self._cfg.twisting.max_offsets, device=self._device, dtype=torch.float64, return_rots=False)

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for i, t_2 in enumerate(ts[1:]):
            if verbose: # and i % 1 == 0:
                print(f'{i=}, t={t_1.item():.2f}')
                print(torch.cuda.mem_get_info(trans_0.device), torch.cuda.memory_allocated(trans_0.device))
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1
            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['so3_t'] = batch['t']
            batch['r3_t'] = batch['t']
            d_t = t_2 - t_1

            use_twisting = motif_scaffolding and self._cfg.twisting.use and t_1 >= self._cfg.twisting.t_min

            if use_twisting: # Reconstruction guidance
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch['r3_t'] #TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(trans_t_1, rotmats_t_1, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj)

            else:
                with torch.no_grad():
                    model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                if motif_scaffolding:
                    batch['trans_sc'] = (
                        pred_trans_1 * diffuse_mask[..., None]
                        + trans_1 * (1 - diffuse_mask[..., None])
                    )
                    batch['rotmats_sc'] = (
                        pred_rotmats_1 * diffuse_mask[..., None, None]
                        + rotmats_1 * (1 - diffuse_mask[..., None, None])
                    )
                else:
                    batch['trans_sc'] = pred_trans_1
                    batch['rotmats_sc'] = pred_rotmats_1

            # Take reverse step
            
            if not self._cfg.inference_time_scaling.use:
                trans_t_2 = self._trans_euler_step(
                    d_t, t_1, pred_trans_1, trans_t_1)
                
            if self._cfg.inference_time_scaling.use:
                with torch.inference_mode(False):
                    trans_pred = pred_trans_1.clone().detach().requires_grad_(True)
                    within_clash_loss = compute_within_clash_loss(
                                        model_out['all_atom_preds']['positions'][-1],
                                        batch['atom14_gt_exists'],
                                        batch["interface_mask"],
                                        batch['aatype'])
                    inter_clash_loss = compute_all_atom_clash_loss(
                        model_out['all_atom_preds']['positions'][-1],
                        batch['atom14_gt_exists'],
                        batch['res_idx'],
                        batch['residx_atom14_to_atom37'],
                        interface_mask=batch['interface_mask']
                    )
                    clash_loss = within_clash_loss + inter_clash_loss
                    grad = torch.autograd.grad(clash_loss, trans_pred)[0]  # shape: (B, N, 3)

                    # Guidance 적용
                    scale = self._cfg.vdw_guidance.scale
                    trans_t_2 = trans_t_2 - scale * grad * d_t

            if trans_potential is not None:
                with torch.inference_mode(False):
                    grad_pred_trans_1 = pred_trans_1.clone().detach().requires_grad_(True)
                    pred_trans_potential = autograd.grad(outputs=trans_potential(grad_pred_trans_1), inputs=grad_pred_trans_1)[0]
                if self._trans_cfg.potential_t_scaling:
                    trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                else:
                    trans_t_2 -= pred_trans_potential * d_t
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            if motif_scaffolding and not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        
        if rollout:
            with torch.inference_mode(False):
                model_out = model(batch)
        else:
            with torch.no_grad():
                model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_positions_14 = model_out['all_atom_preds']['positions'][-1]
        contact_map = model_out['pair_outputs'][-1]

        input_for_confidence = model_out['input_for_confidence']
        prmsd = torch.zeros(batch['diffuse_mask'].shape[0], batch['diffuse_mask'].shape[1], device=batch['diffuse_mask'].device)
        prmsd_final = prmsd

        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, batch["res_mask"])
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, batch["res_mask"])
        # if not (prmsd == 0).all():
        #     prmsd_final = compute_prmsd(prmsd, batch['diffuse_mask'])
        #     print(f'prmsd_final_val : {torch.max(prmsd_final)}')
        # else:
        #     prmsd_final = prmsd 
        if not save_all_repr:
            return atom37_traj, clean_atom37_traj, pred_positions_14, prmsd_final, prmsd, contact_map, pred_trans_1, pred_rotmats_1, input_for_confidence
        else:
            all_input_for_confidence = {
                "atom14_gt_positions": batch["atom14_gt_positions"],
                "atom14_gt_exists": batch["atom14_gt_exists"],
                "diffuse_mask": diffuse_mask,
                "pred_positions": model_out['all_atom_preds']['positions'],
                "input_for_confidence": input_for_confidence
            }
            return atom37_traj, clean_atom37_traj, pred_positions_14, prmsd_final, prmsd, contact_map, pred_trans_1, pred_rotmats_1, all_input_for_confidence
    
    def guidance(self, trans_t, rotmats_t, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj):
        # Select motif
        motif_mask = motif_mask.clone()
        trans_pred = model_out['pred_trans'][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out['pred_rotmats'][:, motif_mask]  # [B, motif_res, 3, 3]

        # Proposal for marginalising motif rotation
        F = twisting.motif_rots_vec_F(trans_motif, R_motif, self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, device=self._device, dtype=torch.float32)

        # Estimate p(motif|predicted_motif)
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(R_pred, trans_pred, R_motif, trans_motif, Log_delta_R, delta_x, None, None, None, F, twist_potential_rot=self._cfg.twisting.potential_rot, twist_potential_trans=self._cfg.twisting.potential_trans)

        with torch.no_grad():
            # Choose scaling
            t_trans = t
            t_so3 = t
            if self._cfg.twisting.scale_w_t == 'ot':
                var_trans = ((1 - t_trans) / t_trans)[:, None]
                var_rot = ((1 - t_so3) / t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'linear':
                var_trans = (1 - t)[:, None]
                var_rot = (1 - t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'constant':
                num_batch = trans_pred.shape[0]
                var_trans = torch.ones((num_batch, 1, 1)).to(R_pred.device)
                var_rot = torch.ones((num_batch, 1, 1, 1)).to(R_pred.device)
            var_trans = var_trans + self._cfg.twisting.obs_noise ** 2
            var_rot = var_rot + self._cfg.twisting.obs_noise ** 2

            trans_scale_t = self._cfg.twisting.scale / var_trans
            rot_scale_t = self._cfg.twisting.scale / var_rot

            # Compute update
            trans_t, rotmats_t = twisting.step(trans_t, rotmats_t, grad_x_log_p_motif, grad_Log_delta_R, d_t, trans_scale_t, rot_scale_t, self._cfg.twisting.update_trans, self._cfg.twisting.update_rot)

        # delete unsused arrays to prevent from any memory leak
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)

        return trans_t, rotmats_t, logs_traj
