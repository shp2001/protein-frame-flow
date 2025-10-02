import torch
from data import utils as du
from data import all_atom
import copy

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _r_diffuse_mask(r_t, r_1, atom_diffuse_mask):
    return r_t * atom_diffuse_mask[..., None] + r_1 * (1 - atom_diffuse_mask[..., None])

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
