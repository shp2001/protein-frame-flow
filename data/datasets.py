import abc
import numpy as np
import pandas as pd
import logging
import torch
from collections import defaultdict

from torch.utils.data import Dataset
from data import utils as du
from data import residue_constants as rc 

from openfold.data import data_transforms
from openfold.utils import rigid_utils
import json 

from data.motif_index import load_loop_file, load_monomer_mask, load_polymer_mask, crop_antigen, crop_general_protein, provide_anchor

# def _rog_filter(df, quantile):
#     y_quant = pd.pivot_table(
#         df,
#         values='radius_gyration', 
#         index='modeled_seq_len',
#         aggfunc=lambda x: np.quantile(x, quantile)
#     )
#     x_quant = y_quant.index.to_numpy()
#     y_quant = y_quant.radius_gyration.to_numpy()

#     # Fit polynomial regressor
#     poly = PolynomialFeatures(degree=4, include_bias=True)
#     poly_features = poly.fit_transform(x_quant[:, None])
#     poly_reg_model = LinearRegression()
#     poly_reg_model.fit(poly_features, y_quant)

#     # Calculate cutoff for all sequence lengths
#     max_len = df.modeled_seq_len.max()
#     pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
#     # Add a little more.
#     pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

#     row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x-1])
#     return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.seq_len >= min_res)
        & (data_csv.seq_len <= max_res)
    ]


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
    chain_feats = data_transforms.atom37_to_torsion_angles(chain_feats)
    chain_feats = data_transforms.get_chi_angles(chain_feats)
    chain_feats = data_transforms.get_backbone_frames(chain_feats)
    chain_feats['pseudo_beta'] = data_transforms.pseudo_beta_fn(
        chain_feats['aatype'],
        chain_feats['all_atom_positions'],
        None
        )
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()
    res_mask[chain_feats['aatype'] == 20] = 0
    chain_idx = torch.tensor(processed_feats['chain_index'])
    residue_index = torch.tensor(processed_feats['residue_index'])

    return {
        'res_plddt': torch.tensor(res_plddt),
        'aatype': chain_feats['aatype'],
        'chain_index': chain_feats['chain_index'],
        'res_mask': res_mask,
        'chain_idx': chain_idx,
        'residue_index': residue_index,
        'scaffold_idx': scaffold_idx,
        'chain_seq_list': chain_seq_list,
        'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        'alt_torsion_angles_sin_cos': chain_feats['alt_torsion_angles_sin_cos'],
        'torsion_angles_mask': chain_feats['torsion_angles_mask'],
        'chi_angles_sin_cos': chain_feats['chi_angles_sin_cos'],
        'chi_mask': chain_feats['chi_mask'],
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'], # (L, 14, 3)
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'], # require centering 
        'atom14_alt_gt_positions': chain_feats['atom14_alt_gt_positions'], # require centering 
        'atom14_alt_gt_exists': chain_feats['atom14_alt_gt_exists'],
        'atom14_atom_is_ambiguous': chain_feats['atom14_atom_is_ambiguous'],
        'backbone_rigid_mask': chain_feats['backbone_rigid_mask'],
        'rigidgroups_gt_frames': chain_feats['rigidgroups_gt_frames'], # require centering  (L, 8, 4, 4)
        'rigidgroups_gt_exists': chain_feats['rigidgroups_gt_exists'],
        'rigidgroups_alt_gt_frames': chain_feats['rigidgroups_alt_gt_frames'], # require centering (L, 8, 4, 4)
    }


def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path):
    with open(cluster_path, 'r') as f:
        cluster_dict = json.load(f)
    
    pdb_to_cluster = {}
    for cluster_id, pdb_ids in cluster_dict.items():
        for pdb_id in pdb_ids:
            pdb_to_cluster[pdb_id] = cluster_id
    
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.current_epoch = None

        if is_training:
            self.raw_csv = pd.read_csv(self.dataset_cfg.train_csv_path)
        else:
            self.raw_csv = pd.read_csv(self.dataset_cfg.valid_csv_path)
        
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        
    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            self.csv = data_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row, idx):
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

        if csv_row['mode'] == 'nanobody':
            cdr_types = ['h1', 'h2', 'h3']
            for cdr in cdr_types:
                scaffold_idx[f'{cdr}_start'] = int(csv_row[f'{cdr}_start'])
                scaffold_idx[f'{cdr}_end'] = int(csv_row[f'{cdr}_end'])

        if csv_row['mode'] == 'general':
            loop_info_file = csv_row['loop_info_dir']
            loop_start, loop_end, masked_chain, first_chain_len = load_loop_file(loop_info_file, seed=self.current_epoch + idx)
            scaffold_idx[f'loop_start'] = loop_start
            scaffold_idx[f'loop_end'] = loop_end
        
        if csv_row['mode'] == 'monomer':
            mask_info_file = csv_row['mask_info_file']
            loop_start, loop_end = load_monomer_mask(mask_info_file, seq_len, seed=self.current_epoch + idx)
            scaffold_idx[f'loop_start'] = loop_start
            scaffold_idx[f'loop_end'] = loop_end

        if csv_row['mode'] == 'polymer':
            mask_info_file = csv_row['mask_info_file']
            interface_start, interface_end = load_polymer_mask(mask_info_file, seed=self.current_epoch + idx)
            scaffold_idx[f'loop_start'] = interface_start
            scaffold_idx[f'loop_end'] = interface_end

        processed_row = _process_csv_row(path, raw_path, scaffold_idx)
        processed_row['masked_chain'] = masked_chain
        processed_row['first_chain_len'] = first_chain_len
        processed_row['raw_path'] = raw_path
        processed_row['mode'] = csv_row['mode']
        return processed_row
    
    def _sample_scaffold_mask(self, batch, rng):
        aatype = batch['aatype']
        mode = batch['mode']
        num_res = aatype.shape[0]
        scaffold_idx = batch['scaffold_idx']
        scaffold_mask = torch.zeros(num_res)

        if mode == 'general' or mode == 'polymer' or mode == 'monomer': # general loop PPI
            loop_indices = []
            for scf, idx in scaffold_idx.items():
                loop_indices.append(idx)
            loop_indices = sorted(loop_indices)

            scaffold_mask[loop_indices[0]:loop_indices[1]+1] = 1.0

        if mode == 'ab': # antibody-antigen
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(6):
                scaffold_mask[cdr_indices[2*i]:cdr_indices[2*i+1]+1] = 1.0

        if mode == 'nanobody': # Nanobody-antigen
            cdr_indices = []
            for scf, idx in scaffold_idx.items():
                cdr_indices.append(idx)
            cdr_indices = sorted(cdr_indices)
            for i in range(3):
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
        csv_row = self.csv.iloc[row_idx]
        chain_feats = self.process_csv_row(csv_row, row_idx)
        feats = chain_feats.copy()
        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['loop_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':
            rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
            feats['rotmats_1'] = torch.tensor(rigids_1.get_rots().get_rot_mats(), device=rigids_1.device)
            feats['trans_1'] = torch.tensor(rigids_1.get_trans(), device=rigids_1.device)

            rng = self._rng if self.is_training else np.random.default_rng(seed=123)
            self.setup_inpainting(feats, rng)

            # modify loop mask (if it is terminal mask, exclude end residue to provide an anchor)
            feats['loop_mask'] = provide_anchor(
                feats['loop_mask'], 
                feats['res_mask'], 
                feats['chain_index'],
                feats['mode']
                ).to(torch.long)
            
            # make diffuse_mask
            # if sample is monomer -> diffuse_mask = loop_mask
            # if sample is polymer -> diffuse_mask is whole chains which have masked loops
             
            chain_len_list = [len(seq) for seq in feats['chain_seq_list']]
            asym_id = []
            for i, chain_len in enumerate(chain_len_list):
                for _ in range(chain_len):
                    asym_id.append(i)
            asym_id = torch.tensor(asym_id, device=feats['loop_mask'].device)
            masked_chain = asym_id[feats['loop_mask'] == 1].unique()
            diffuse_mask = torch.isin(asym_id, masked_chain).to(torch.long)
            feats['diffuse_mask'] = diffuse_mask

            # create crop_idx for cropping 
            mode = feats['mode']
            
            if mode not in ['ab', 'nanobody', 'general', 'monomer', 'polymer']:
                raise ValueError('Mode should be one of [ab, nanobody, general, monomer, polymer]')

            if mode == 'ab':
                feats['crop_idx'] = crop_antigen(
                    feats['trans_1'],
                    cdr_mask=feats['loop_mask'],
                    nan_mask=feats['res_mask'],
                    max_len=self.dataset_cfg.ab_max_num_res,
                    seq_list=feats['chain_seq_list'],
                    crop_ab=self.dataset_cfg.crop_ab,
                    mode=mode,
                    )
            if mode == 'nanobody':
                feats['crop_idx'] = crop_antigen(
                    feats['trans_1'],
                    cdr_mask=feats['loop_mask'],
                    nan_mask=feats['res_mask'],
                    max_len=self.dataset_cfg.ab_max_num_res,
                    seq_list=feats['chain_seq_list'],
                    crop_ab=self.dataset_cfg.crop_ab,
                    mode=mode,
                    )   
            if mode == 'general' or mode == 'polymer' or mode == 'monomer':
                feats['crop_idx'] = crop_general_protein(
                    feats['trans_1'],
                    loop_mask=feats['loop_mask'],
                    nan_mask=feats['res_mask'],
                    max_len=self.dataset_cfg.general_max_num_res,
                    seq_list=feats['chain_seq_list'],
                    )

        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['loop_mask'] = feats['loop_mask'].int()
        
        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx

        return feats


class ScopeDataset(BaseDataset):

    def _filter_metadata(self, raw_csv):
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv,
            filter_cfg.min_num_res,
            filter_cfg.max_num_res
        )
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv


class PdbDataset(BaseDataset):

    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self.current_epoch = 0
        # Process clusters
        if is_training:
            self.raw_csv = pd.read_csv(self.dataset_cfg.train_csv_path)
            self._pdb_to_cluster = _read_clusters(self._dataset_cfg.train_cluster_path)
        else:
            self.raw_csv = pd.read_csv(self.dataset_cfg.valid_csv_path)
            self._pdb_to_cluster = _read_clusters(self._dataset_cfg.valid_cluster_path)

        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_len', ascending=False)

        
        self._max_cluster = len(self._pdb_to_cluster.values())
        self._missing_pdbs = 0
        def cluster_lookup(pdb):
            if pdb not in list(self._pdb_to_cluster.keys()):
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]
        metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
        self._create_split(metadata_csv)
        self._all_clusters = dict(
            enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv

        # if self._is_training:
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)

        return data_csv
