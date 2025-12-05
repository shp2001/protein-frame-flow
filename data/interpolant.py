import torch
from data import utils as du
from data import all_atom
import copy

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _r_diffuse_mask(r_t, r_1, atom_diffuse_mask):
    return r_t * atom_diffuse_mask[..., None] + r_1 * (1 - atom_diffuse_mask[..., None])

import torch

def axis_angle_to_matrix_batched(axis, angle):
    """
    Rodrigues' rotation formula (Batched)
    axis: (B, 3)
    angle: (B, 1)
    return: (B, 3, 3)
    """
    B = axis.shape[0]
    device = axis.device
    dtype = axis.dtype
    
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    
    # Skew-symmetric matrix K
    K = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1)
    ], dim=1) # (B, 3, 3)
    
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
    
    sin_a = torch.sin(angle).view(B, 1, 1)
    cos_a = torch.cos(angle).view(B, 1, 1)
    
    # R = I + sin(theta)K + (1-cos(theta))K^2
    R = I + sin_a * K + (1 - cos_a) * torch.matmul(K, K)
    return R

def apply_global_rigid_transform(trans, diffuse_mask, translation_scale=2.0, rotation_scale=0.3):
    """
    trans: shape (B, L, 3)
    diffuse_mask: shape (B, L) or (L,)
    """
    coords = trans # (B, L, 3)
    B, L, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # ---------------------------
    # 1) Mask Preparation (Broadcasting)
    # ---------------------------
    if diffuse_mask.dim() == 1:
        mask_bool = diffuse_mask.view(1, L, 1).expand(B, L, 1).bool()
    else:
        mask_bool = diffuse_mask.view(B, L, 1).bool()

    mask_float = mask_bool.float()

    # ---------------------------
    # 2) Center of Mass (Batch-wise)
    # ---------------------------
    # (B, L, 3) * (B, L, 1) -> sum -> (B, 3)
    masked_sum = (coords * mask_float).sum(dim=1)
    mask_count = mask_float.sum(dim=1) # (B, 1)
    mask_count = torch.clamp(mask_count, min=1.0) # 0 나누기 방지
    
    center = masked_sum / mask_count
    center = center.view(B, 1, 3) # (B, 1, 3)

    # ---------------------------
    # 3) Batch Random Rotation & Translation
    # ---------------------------
    if rotation_scale > 1e-6:
        # Axis: (B, 3)
        rand_axis = torch.randn(B, 3, device=device, dtype=dtype)
        rand_axis = rand_axis / (torch.norm(rand_axis, dim=1, keepdim=True) + 1e-6)
        
        # Angle: (B, 1)
        rand_angle = (torch.rand(B, 1, device=device, dtype=dtype) * 2 - 1) * rotation_scale
        
        R = axis_angle_to_matrix_batched(rand_axis, rand_angle) # (B, 3, 3)
    else:
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)

    t = torch.randn(B, 3, device=device, dtype=dtype) * translation_scale
    t = t.view(B, 1, 3)

    # ---------------------------
    # 4) Apply Transform
    # ---------------------------
    # R^T (Transpose for multiplying on the right)
    R_T = R.transpose(1, 2) # (B, 3, 3)
    
    # Effective Translation: t_eff = C - C@R^T + t
    # center: (B, 1, 3)
    center_rotated = torch.matmul(center, R_T)
    t_effective = center - center_rotated + t # (B, 1, 3)
    
    # 전체 좌표 변환: (B, L, 3) @ (B, 3, 3) + (B, 1, 3)
    rotated_all = torch.matmul(coords, R_T)
    transformed_all = rotated_all + t_effective

    # mask_bool: (B, L, 1) -> 자동으로 (B, L, 3)으로 브로드캐스팅되어 조건 적용
    new_coords = torch.where(mask_bool, transformed_all, coords)

    return new_coords
            
class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

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

        return noisy_batch
    

    def _r3_vector_field(self, t, r_1, r_t):
        return (r_1 - r_t) / (1 - t)

    def _r3_euler_step(self, d_t, t, r_1, r_t):
        assert d_t > 0
        trans_vf = self._r3_vector_field(t, r_1, r_t)
        return r_t + trans_vf * d_t

    def sample(
            self,
            model,
            batch,
            s_init, 
            s_trunk, 
            z_trunk,
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
                model_out = model.get_structure(batch, s_init, s_trunk, z_trunk)

            # Process model output.
            pred_r_1 = model_out['pred_r_1']
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))

            # Take reverse step
            r_t_2 = self._r3_euler_step(
                d_t, t_1, pred_r_1, r_t_1)
            r_t_2 = _r_diffuse_mask(r_t_2, r_1, atom_diffuse_mask)
            prot_traj.append(r_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        r_t_1 = prot_traj[-1]
        batch['r_t'] = r_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1

        with torch.no_grad():
            model_out = model.get_structure(batch, s_init, s_trunk, z_trunk)
                
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_r_1 = model_out['pred_r_1']
        pred_positions_14 = model_out['pred_r_1_unflatten']
        clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))
        prot_traj.append(pred_r_1)

        # Convert trajectories to atom37.
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, batch["res_mask"])

        return prot_traj, clean_atom37_traj, pred_positions_14, pred_trans_1
