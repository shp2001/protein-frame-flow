# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Library for parsing different data structures.
Code adapted from Openfold protein.py.
"""
from Bio.PDB.Chain import Chain
import numpy as np

from data import residue_constants
from data import protein
from openfold.data import data_transforms

Protein = protein.Protein


def add_openfold_data_transforms(
    chain_information
):
    # this is a hack because openfold will put an aatype as unknown even if the residue
    # is known but the structural information is missing. Without this hack the
    # atom14 mask and other various quantities will be incorrect
    aatype = chain_information["aatype"].clone()

    chain_information["aatype"][~chain_information["seq_mask"].bool()] = (
        residue_constants.restype_order_with_x["X"]
    )

    atom14_transforms = [
        data_transforms.make_atom14_masks,
        data_transforms.make_atom14_positions,
    ]

    for transform in atom14_transforms:
        chain_information = transform(chain_information)

    rest_of_transforms = [
        data_transforms.atom37_to_frames,
        data_transforms.atom37_to_torsion_angles(""),
        data_transforms.make_pseudo_beta(""),
        data_transforms.get_backbone_frames,
        data_transforms.get_chi_angles,
    ]

    chain_information["all_atom_positions"] = chain_information[
        "all_atom_positions"
    ].double()
    for transform in rest_of_transforms:
        chain_information = transform(chain_information)

    to_cast_to_float = [
        "all_atom_positions",
        "atom14_gt_positions",
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
        "pseudo_beta",
    ]
    for tensor in to_cast_to_float:
        chain_information[tensor] = chain_information[tensor].float()

    # return aatype to be non-masked version
    chain_information["aatype"] = aatype
    return chain_information

def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.
    
    Forked from alphafold.common.protein.from_pdb_string
    
    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.
    
    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.
    
    Took out lines 110-112 since that would mess up CDR numbering.
    
    Args:
        chain: Instance of Biopython's chain class.
    
    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]
                          ] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors))