from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from torch import autograd
from motif_scaffolding import twisting
from models.loss import clash_potential

from openfold.utils import rigid_utils

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _r_diffuse_mask(r_t, r_1, atom_diffuse_mask):
    return r_t * atom_diffuse_mask[..., None] + r_1 * (1 - atom_diffuse_mask[..., None])

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

    def sample_t(self, num_batch, mode):
        if mode not in ["uniform", "beta_mixed", "beta_revese_mixed", "logit_normal"]:
            raise ValueError(f'mode ({mode}) should be one of uniform, beta_mixed, beta_revese_mixed or logit_normal')
        
        if mode == 'uniform':
            t = torch.rand(num_batch, device=self._device)
            return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t
        
        if mode == 'beta_mixed':
            probs = torch.rand(num_batch, device=self._device)
            unif_mask = probs < 0.02
            beta_mask = ~unif_mask

            samples = torch.empty(num_batch, device=self._device)
            samples[unif_mask] = torch.rand(unif_mask.sum(), device=self._device) * (1 - 2 * self._cfg.min_t) + self._cfg.min_t
            beta_samples = torch.distributions.Beta(1.9, 1.0).sample((beta_mask.sum(),)).to(self._device)
            samples[beta_mask] = torch.clamp(beta_samples, min=self._cfg.min_t, max=1 - self._cfg.min_t)
            return samples 

        if mode == 'beta_revese_mixed':
            probs = torch.rand(num_batch, device=self._device)
            unif_mask = probs < 0.02
            beta_mask = ~unif_mask

            samples = torch.empty(num_batch, device=self._device)
            samples[unif_mask] = torch.rand(unif_mask.sum(), device=self._device) * (1 - 2 * self._cfg.min_t) + self._cfg.min_t
            beta_samples = torch.distributions.Beta(1.0, 1.9).sample((beta_mask.sum(),)).to(self._device)
            samples[beta_mask] = torch.clamp(beta_samples, min=self._cfg.min_t, max=1 - self._cfg.min_t)
            return samples 

        if mode == 'logit_normal':
            z = torch.randn(num_batch, device=self._device)
            z = torch.sigmoid(z)
            return torch.clamp(z, min=self._cfg.min_t)

        
    def _corrupt_r(self, r_1, t, atom_diffuse_mask):
        r_0 = _centered_gaussian(*atom_diffuse_mask.shape, self._device)
        r_0 = r_0 * du.NM_TO_ANG_SCALE
        r_0 = _r_diffuse_mask(r_0, r_1, atom_diffuse_mask)
        r_t = (1 - t[..., None]) * r_0 + t[..., None] * r_1
        
        return r_t
    
    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        atom_diffuse_mask = batch['atom_diffuse_mask']
        r_1 = batch['r_1']
        num_batch, _ = atom_diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch, self._cfg.sample_t_mode)[:, None]
        noisy_batch['t'] = t

        r_t = self._corrupt_r(r_1, t, atom_diffuse_mask)
        
        noisy_batch['r_t'] = r_t
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
            model,
            batch,
            num_timesteps=None,
            verbose=False,
        ):


        diffuse_mask = batch['diffuse_mask']
        atom_diffuse_mask = batch['atom_diffuse_mask']
        num_batch, num_atom = atom_diffuse_mask.shape

        motif_mask = ~diffuse_mask.bool().squeeze(0)
        r_1 = batch['r_1']

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up initial prior samples

        # random_trans = _random_translations(trans_1.shape[0], self._cfg.max_shift, self._device)
        # random_rot = _random_rotation_matrices(trans_1.shape[0], self._cfg.max_angle_deg, device=self._device)

        r_0 = _centered_gaussian(num_batch, num_atom, self._device)
        r_0 = r_0 * du.NM_TO_ANG_SCALE
        r_0 = _r_diffuse_mask(r_0, r_1, atom_diffuse_mask)

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [r_0]
        clean_traj = []
        for i, t_2 in enumerate(ts[1:]):
            if verbose: # and i % 1 == 0:
                print(f'{i=}, t={t_1.item():.2f}')
                print(torch.cuda.mem_get_info(r_0.device), torch.cuda.memory_allocated(r_0.device))
            # Run model.
            r_t_1 = prot_traj[-1]
            batch['r_t'] = r_t_1
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1

            d_t = t_2 - t_1

            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            pred_r_1 = model_out['pred_r_1']

            clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
            
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1
                batch['rotmats_sc'] = pred_rotmats_1

            # Take reverse step
            r_t_2 = self._trans_euler_step(
                d_t, t_1, pred_r_1, r_t_1)

            prot_traj.append(r_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        r_t_1 = prot_traj[-1]
        batch['r_t'] = r_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1

        with torch.no_grad():
            model_out = model(batch)
                
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_r_1 = model_out['pred_r_1']

        pred_positions_14 = model_out['all_atom_preds']['positions'][-1]
        pair_outputs = model_out['pair_outputs']
        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        prot_traj.append(pred_r_1)

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, batch["res_mask"])
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, batch["res_mask"])

        prmsd = torch.zeros(batch['diffuse_mask'].shape[0], batch['diffuse_mask'].shape[1], device=batch['diffuse_mask'].device)
        prmsd_final = prmsd

        return prot_traj, clean_atom37_traj, pred_positions_14, prmsd_final, prmsd, pred_trans_1, pred_rotmats_1, pair_outputs

    
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
