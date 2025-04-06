import abc
import numpy as np
import pandas as pd
import logging
import torch


from torch.utils.data import Dataset
from data import utils as du


from openfold.data import data_transforms
from openfold.utils import rigid_utils
import json 

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from data.motif_index import load_loop_file

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
    chain_seq_list = []
    pdb_file = raw_path

    p = PDBParser()
    structure = p.get_structure(
        'protein',
        pdb_file,
    )

    for chain in structure.get_chains():
        pdb_seq = "".join([seq1(r.get_resname()) for r in chain.get_residues()])
        chain_seq_list.append(pdb_seq)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
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
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    chain_idx = torch.tensor(processed_feats['chain_index'])
    res_idx = processed_feats['residue_index']

    chain_feats['pseudo_beta'] = data_transforms.pseudo_beta_fn(
                                                                chain_feats['aatype'],
                                                                chain_feats['all_atom_positions'],
                                                                None)
    return {
        'res_plddt': torch.tensor(res_plddt),
        'aatype': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1, # require centering 
        'res_mask': res_mask,
        'chain_idx': chain_idx,
        'res_idx': res_idx,
        'scaffold_idx': scaffold_idx,
        'chain_seq_list': chain_seq_list,
        'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        'alt_torsion_angles_sin_cos': chain_feats['alt_torsion_angles_sin_cos'],
        'torsion_angles_mask': chain_feats['torsion_angles_mask'],
        'chi_angles_sin_cos': chain_feats['chi_angles_sin_cos'],
        'chi_mask': chain_feats['chi_mask'],
        'atom14_gt_exists': chain_feats['atom14_gt_exists'],
        'atom14_gt_positions': chain_feats['atom14_gt_positions'], # require centering 
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'pseudo_beta': chain_feats['pseudo_beta'], # require centering 
        'atom14_alt_gt_positions': chain_feats['atom14_alt_gt_positions'], # require centering 
        'atom14_alt_gt_exists': chain_feats['atom14_alt_gt_exists'],
        'atom14_atom_is_ambiguous': chain_feats['atom14_atom_is_ambiguous'],
        'backbone_rigid_tensor': chain_feats['backbone_rigid_tensor'], # require centering  (L, 4, 4)
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

        if is_training:
            self.raw_csv = pd.read_csv(self.dataset_cfg.train_csv_path)
        else:
            self.raw_csv = pd.read_csv(self.dataset_cfg.valid_csv_path)
        
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
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

            loop_start, loop_end, masked_chain, first_chain_len = load_loop_file(loop_info_file)
            scaffold_idx[f'loop_start'] = loop_start
            scaffold_idx[f'loop_end'] = loop_end

        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        
        processed_row = _process_csv_row(path, raw_path, scaffold_idx)
        processed_row['masked_chain'] = masked_chain
        processed_row['first_chain_len'] = first_chain_len
        processed_row['raw_path'] = raw_path
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
        diffuse_mask = self._sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            diffuse_mask = diffuse_mask * feats['plddt_mask']
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats['diffuse_mask'] = diffuse_mask
    
    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['diffuse_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':

            rng = self._rng if self.is_training else np.random.default_rng(seed=123)
            self.setup_inpainting(feats, rng)

            # Center based on motif locations
            motif_mask = 1 - feats['diffuse_mask']
            trans_1 = feats['trans_1']
            motif_1 = trans_1 * motif_mask[:, None]
            motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
            trans_1 -= motif_com[None, :]
            feats['trans_1'] = trans_1
            feats['atom14_gt_positions'] -= motif_com[None, :]
            feats['pseudo_beta'] -= motif_com[None, :]
            feats['atom14_alt_gt_positions'] -= motif_com[None, :]
            feats['backbone_rigid_tensor'][:, :3, 3] -= motif_com[None, :]
            feats['rigidgroups_gt_frames'][:, :, :3, 3] -= motif_com[None, None, :]
            feats['rigidgroups_alt_gt_frames'][:, :, :3, 3] -= motif_com[None, None, :]

            print(f'motif_com: {motif_com}')
            print(f'trans_1: {trans_1[0]}')
        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['diffuse_mask'] = feats['diffuse_mask'].int()
        
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
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

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

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)

        return data_csv
