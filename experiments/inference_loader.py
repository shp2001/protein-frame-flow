import abc
import numpy as np
import pandas as pd
import logging
import torch
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from data import utils as du


from openfold.data import data_transforms
from openfold.utils import rigid_utils
import json 

from data.motif_index import get_relpos_input, crop_antigen, load_loop_file, load_monomer_mask, load_polymer_mask, crop_general_protein, provide_anchor, get_ag_hotspot
from data import residue_constants as rc

from itertools import accumulate
import bisect

from data import featurizer
from torch.utils.data import SequentialSampler

def _process_csv_row(processed_file_path, raw_path, scaffold_idx):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # make chain sequence list (for the multimer relpos embedding)
    int_to_aa = {i: restype for restype, i in rc.restype_order_with_x.items()}
    aatypes = processed_feats["aatype"]             # [L]
    chain_indices = processed_feats["chain_index"]  # [L]

    chain_seqs = defaultdict(list)
    chain_order = []

    for aa_int, chain_id in zip(aatypes, chain_indices):
        aa_letter = int_to_aa.get(int(aa_int), "X")
        chain_seqs[chain_id].append(aa_letter)
        if chain_id not in chain_order:
            chain_order.append(chain_id)

    chain_seq_list = ["".join(chain_seqs[chain_id]) for chain_id in chain_order]

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'chain_index': torch.tensor(processed_feats['chain_index']),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).float(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).float(),
        'seq_mask': torch.tensor(processed_feats['bb_mask']).int()
    }
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats['pseudo_beta'] = data_transforms.pseudo_beta_fn(
                                                                chain_feats['aatype'],
                                                                chain_feats['all_atom_positions'],
                                                                None)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    chain_idx = torch.tensor(processed_feats['chain_index'])
    residue_index = torch.tensor(processed_feats['residue_index'])

    return {
        'res_plddt': torch.tensor(res_plddt),
        'aatype': chain_feats['aatype'],
        'chain_index': chain_feats['chain_index'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'chain_idx': chain_idx,
        'residue_index': residue_index,
        'scaffold_idx': scaffold_idx,
        'chain_seq_list': chain_seq_list,
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'],
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'],
    }


class BaseDataset(Dataset):
    def __init__(
            self,
            inf_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._inf_cfg = inf_cfg
        self._inference_cfg = inf_cfg.inference
        self.task = task

        self.n_samples = self._inference_cfg.samples.samples_per_target

        self.raw_csv = pd.read_csv(self._inference_cfg.samples.csv_path)

        metadata_csv = self.raw_csv
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=123)

        all_sample_ids = []
        for row_id in range(self.csv.shape[0]):
            target_row = self.csv.iloc[row_id]
            for sample_id in range(self.n_samples):
                sample_id = torch.tensor(sample_id)
                all_sample_ids.append((target_row, sample_id))

        self._all_sample_ids = all_sample_ids
        
    @property
    def is_training(self):
        return self._is_training

    
    def __len__(self):
        return len(self._all_sample_ids)
    
    def _create_split(self, data_csv):
        # Training or validation specific logic.
        self.csv = data_csv
        self._log.info(
            f'Validation: {len(self.csv)} examples')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']
        raw_path = csv_row['raw_path']
        seq_len = csv_row['seq_len']
        
        masked_chain = None
        first_chain_len = None

        scaffold_idx = {}

        if csv_row['mode'] == 'ab':
            cdr_types = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3']
            for cdr in cdr_types:
                scaffold_idx[f'{cdr}_start'] = int(csv_row[f'{cdr}_start'])
                scaffold_idx[f'{cdr}_end'] = int(csv_row[f'{cdr}_end'])

        if csv_row['mode'] == 'general':
            loop_info_file = csv_row['loop_info_dir']

            loop_start, loop_end, masked_chain, first_chain_len = load_loop_file(loop_info_file, seed=None)
            scaffold_idx[f'loop_start'] = loop_start
            scaffold_idx[f'loop_end'] = loop_end

        if csv_row['mode'] == 'general':
            loop_info_file = csv_row['loop_info_dir']
            loop_start, loop_end, masked_chain, first_chain_len = load_loop_file(loop_info_file, seed=123)
            scaffold_idx[f'loop_start'] = loop_start
            scaffold_idx[f'loop_end'] = loop_end
        
        if csv_row['mode'] == 'monomer':
            mask_info_file = csv_row['mask_info_file']
            loop_start, loop_end = load_monomer_mask(mask_info_file, seq_len, seed=123)
            scaffold_idx[f'loop_start'] = loop_start
            scaffold_idx[f'loop_end'] = loop_end

        if csv_row['mode'] == 'polymer':
            mask_info_file = csv_row['mask_info_file']
            interface_start, interface_end = load_polymer_mask(mask_info_file, seed=123)
            scaffold_idx[f'loop_start'] = interface_start
            scaffold_idx[f'loop_end'] = interface_end

        # Large protein files are slow to read. Cache them.
        use_cache = True
        if use_cache and path in self._cache:
            return self._cache[path]
        
        processed_row = _process_csv_row(path, raw_path, scaffold_idx)
        processed_row['masked_chain'] = masked_chain
        processed_row['first_chain_len'] = first_chain_len
        processed_row['raw_path'] = raw_path
        processed_row['processed_path'] = path
        processed_row['mode'] = csv_row['mode']
        if use_cache:
            self._cache[path] = processed_row
        
        return processed_row
    
    def _sample_scaffold_mask(self, batch, rng):
        trans_1 = batch['trans_1']
        num_res = trans_1.shape[0]
        scaffold_idx = batch['scaffold_idx']
        scaffold_mask = torch.zeros(num_res)

        if len(scaffold_idx.keys()) == 2: # general loop PPI
            loop_indices = []
            for scf, idx in scaffold_idx.items():
                loop_indices.append(idx)
            loop_indices = sorted(loop_indices)

            scaffold_mask[loop_indices[0]:loop_indices[1]+1] = 1.0

        elif len(scaffold_idx.keys()) == 12: # antibody-antigen
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(6):
                scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1.0

        return scaffold_mask * batch['res_mask']
    
    def setup_inpainting(self, feats, rng):
        loop_mask = self._sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            loop_mask = loop_mask * feats['plddt_mask']
        if torch.sum(loop_mask) < 1:
            # Should only happen rarely.
            loop_mask = torch.ones_like(loop_mask)
        feats['loop_mask'] = loop_mask
    
    def __getitem__(self, row_idx):
        # Process data example.
        csv_row, sample_id = self._all_sample_ids[row_idx]
        feats = self.process_csv_row(csv_row)

        feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['loop_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':

            rng = np.random.default_rng(seed=123)
            self.setup_inpainting(feats, rng)
            feats['loop_mask'] = provide_anchor(feats['loop_mask'], 
                                                   feats['res_mask'], 
                                                   feats['chain_index'],
                                                   feats['mode']).to(torch.long)

        else:
            raise ValueError(f'Unknown task {self.task}')

        # make diffuse_mask
        # if sample is monomer -> diffuse_mask = loop_mask
        # if sample is polymer -> diffuse_mask is whole chains which have masked loops
            
        chain_len_list = [len(seq) for seq in feats['chain_seq_list']]
        if len(chain_len_list) == 1:
            diffuse_mask = feats['loop_mask']
        else:
            asym_id = []
            for i, chain_len in enumerate(chain_len_list):
                for _ in range(chain_len):
                    asym_id.append(i)
            asym_id = torch.tensor(asym_id, device=feats['loop_mask'].device)
            masked_chain = asym_id[feats['loop_mask'] == 1].unique()
            diffuse_mask = torch.isin(asym_id, masked_chain).to(torch.long)
        feats['diffuse_mask'] = diffuse_mask
        
        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx
        feats['sample_id'] = sample_id

        return feats


def collate_fn(batch):
    cropped_batch = []
    for feat in batch:
        mode = feat['mode']
        # crop the feats
        cropped_feat = {}

        if mode =='ab':
            cropped_feat['crop_idx'] = crop_antigen(feat['trans_1'],
                                                    cdr_mask=feat['loop_mask'],
                                                    nan_mask=feat['res_mask'],
                                                    max_len=1000,
                                                    seq_list=feat['chain_seq_list'],
                                                    crop_ab=False
                                                    )
        if mode == 'general' or mode == 'polymer' or mode == 'monomer':
            cropped_feat['crop_idx'] = crop_general_protein(feat['trans_1'],
                            loop_mask=feat['loop_mask'],
                            nan_mask=feat['res_mask'],
                            max_len=256,
                            seq_list=feat['chain_seq_list']
                            )

        not_crop_key = ['crop_idx', 'scaffold_idx', 'chain_seq_list', 'csv_idx', 'masked_chain', 'first_chain_len', 'raw_path', 'processed_path', 'sample_id', 'mode']

        for key in feat.keys():
            if key not in not_crop_key:
                cropped_feat[key] = feat[key][cropped_feat['crop_idx']]

            if key == 'chain_seq_list':
                lengths = [len(s) for s in feat[key]]
                start_positions = list(accumulate([0] + lengths))

                merged = "".join(feat[key])
                cropped_seq_list = [[] for _ in range(len(feat[key]))]

                for idx in cropped_feat['crop_idx']:
                    chain_idx = bisect.bisect_right(start_positions, idx) - 1
                    cropped_seq_list[chain_idx].append(merged[idx])

                cropped_feat[key] = ["".join(chain_seq) for chain_seq in cropped_seq_list]

        # make pair_init (relpos)
        asym_id, entity_id, sym_id = get_relpos_input(cropped_feat['chain_seq_list'])
        cropped_feat['asym_id'] = torch.tensor(asym_id)
        cropped_feat['entity_id'] = torch.tensor(entity_id)
        cropped_feat['sym_id'] = torch.tensor(sym_id)

        cropped_feat['csv_idx'] = feat['csv_idx']
        cropped_feat['crop_idx'] = torch.tensor(cropped_feat['crop_idx'])
        cropped_feat['sample_id'] = torch.tensor(feat['sample_id'], device=feat['aatype'].device)
        del cropped_feat['chain_seq_list']

        cropped_batch.append(cropped_feat)
        

    cropped_batch = {key: [d[key] for d in cropped_batch] for key in cropped_batch[0].keys()}   

    for key in cropped_batch.keys():                
        cropped_batch[key] = torch.stack(cropped_batch[key], dim=0)  


    ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, atom_to_tokatom_idx, ref_pos = featurizer.get_ref_basic_feature(cropped_batch['aatype'], cropped_batch['atom14_gt_exists'], cropped_batch['residue_index'])
    cropped_batch['ref_feature_dict'] = {
        'ref_space_uid': ref_space_uid,
        'ref_element': ref_element,
        'ref_charge': ref_charge,
        'ref_atom_name_chars': ref_atom_name_chars,
        'atom_to_token_idx': atom_to_token_idx,
        'atom_to_tokatom_idx': atom_to_tokatom_idx,
        'ref_pos': ref_pos,
        }

    cropped_batch['ag_hotspot'] = get_ag_hotspot(
        cropped_batch['pseudo_beta'],
        cropped_batch['loop_mask'],
        cropped_batch['diffuse_mask'],
        threshold=8
    )
    # Center based on motif locations
    motif_mask = 1 - cropped_batch['loop_mask'] # (B, L)
    motif_1 = cropped_batch['trans_1'] * motif_mask[..., None] # (B, L, 3)
    motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask, dim=1) + 1)[..., None] # (B, 3)

    cropped_batch["trans_1"] = cropped_batch['trans_1'] - motif_com[:, None, :] # (B, L, 3)
    cropped_batch['atom14_gt_positions'] = cropped_batch['atom14_gt_positions'] - motif_com[:, None, None, :] # (B, L, 14, 3)
    cropped_batch['pseudo_beta'] = cropped_batch['pseudo_beta'] - motif_com[:, None, :] # (B, L, 3)

    cropped_batch['raw_path'] = feat['raw_path']
    cropped_batch['processed_path'] = feat['processed_path']
    cropped_batch['mode'] = feat['mode']
    cropped_batch['original_atom14_gt_positions'] = feat['atom14_gt_positions']
    cropped_batch['original_residx_atom37_to_atom14'] = feat['residx_atom37_to_atom14']
    cropped_batch['original_atom37_atom_exists'] = feat['atom37_atom_exists']
    cropped_batch['original_aatype'] = feat['aatype']
    cropped_batch['original_chain_idx'] = feat['chain_idx']
    cropped_batch['original_diffuse_mask'] = feat['diffuse_mask']

    # create atom diffuse_mask 
    atom_diffuse_mask = cropped_batch['atom14_gt_exists'].clone() # (B, L, 14)
    atom_diffuse_mask[..., :3] *= cropped_batch["diffuse_mask"].unsqueeze(-1)
    cropped_batch['atom_diffuse_mask'] = du.atom_flatten(atom_diffuse_mask, cropped_batch['atom14_gt_exists'])
    cropped_batch['r_1'] = du.atom_flatten(cropped_batch['atom14_gt_positions'], cropped_batch['atom14_gt_exists'])

    # edge mask 
    cropped_batch["edge_mask"] = cropped_batch['res_mask'][:, None] * cropped_batch['res_mask'][:, :, None]

    return cropped_batch


def predict_dataloader(dataset,
                       loader_cfg):
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        sampler=SequentialSampler(dataset),
        prefetch_factor=None if loader_cfg.num_workers == 0 else loader_cfg.prefetch_factor,
        pin_memory=False,
        persistent_workers=True if loader_cfg.num_workers > 0 else False,
        collate_fn=collate_fn
    )