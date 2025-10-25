import torch
import numpy as np 

from data import residue_constants
from scipy.spatial.transform import Rotation

def random_transform(
    points, max_translation=1.0, apply_augmentation=True, centralize=True
) -> np.ndarray:
    """
    Randomly transform a set of 3D points.

    Args:
        points (numpy.ndarray): The points to be transformed, shape=(N, 3)
        max_translation (float): The maximum translation value. Default is 1.0.
        apply_augmentation (bool): Whether to apply random rotation/translation on ref_pos

    Returns:
        numpy.ndarray: The transformed points.
    """
    if centralize:
        points = points - points.mean(axis=0)
    if not apply_augmentation:
        return points
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    R = Rotation.random().as_matrix()
    transformed_points = np.dot(points + translation, R.T)
    return transformed_points

@staticmethod
def atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
    """
    Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
    One-hot encoding of the unique atom names in the reference conformer.
    Each character is encoded as ord(c) − 32, and names are padded to length 4.

    Args:
        atom_name_list (List[str]): A list of atom names.

    Returns:
        torch.Tensor:  A Tensor of character encoded atom names
    """
    onehot_dict = {}
    for index, key in enumerate(range(64)):
        onehot = [0] * 64
        onehot[index] = 1
        onehot_dict[key] = onehot
    # [N_atom, 4, 64]
    mol_encode = []
    for atom_name in atom_names:
        # [4, 64]
        atom_encode = []
        for name_str in atom_name.ljust(3):
            atom_encode.append(onehot_dict[ord(name_str) - 32])
        mol_encode.append(atom_encode)

    return mol_encode

def get_ref_basic_feature(aatype_batch, atom_14_mask_batch, res_indices_batch):
    """
    input
    aatype: (B, N)
    atom14_mask: (B, N)
    res_indices: (B, N)

    return 
    ref_space_uid: tensor[tuple] (N, 2)    (chain_id, residue_index)
    ref_element: tensor (N, 5)
    ref_charge: tensor (N)
    ref_atom_name_chars: tensor (N, 4, 64)
    ref_pos: tensor (B, N, 3) 
    """
    B = aatype_batch.shape[0]

    atom_list = []
    ref_space_uid = []
    ref_element = []
    ref_charge = []
    atom_to_token_idx = [] # residue number를 고려하지 않고 res idx 상에서 몇 번째인지 
    atom_to_tokatom_idx = []  # residue 내 atom index
    ref_pos = []

    aatype = aatype_batch[0]
    res_indices = res_indices_batch[0]
    atom14_mask = atom_14_mask_batch[0]

    for i, restype_int in enumerate(aatype):
        restype1 = residue_constants.restypes_with_x[restype_int]
        restype3 = residue_constants.restype_1to3.get(restype1, "UNK")
        
        if restype3 == "UNK":
            print("There is a UNK in restype")
            continue

        atom_names = residue_constants.restype_name_to_atom14_names[restype3] # atom list 
        atom_coords = residue_constants.atom_positions_ideal[restype3] 

        res_idx = res_indices[i]
        res_coords = []
        local_idx = 0
        for j, atom_name in enumerate(atom_names):
            if atom_name != '' and atom14_mask[i, j] == 1:
                ref_space_uid.append(res_idx)
                atom_list.append(atom_name)
                element = residue_constants.atom_type_to_element[atom_name]
                element_one_hot = residue_constants.element_onehot[element] # numpy array
                
                # ref_element
                ref_element.append(element_one_hot)

                # ref_charge
                charge = residue_constants.atom_type_to_charge[atom_name]
                ref_charge.append(charge)

                # atom_to_token_idx
                atom_to_token_idx.append(i)

                # atom_to_tokatom_idx
                atom_to_tokatom_idx.append(local_idx)
                local_idx += 1 

                # ref_pos & ref_rigid_frame
                coord = None
                for atom in atom_coords:
                    if atom[0] == atom_name:
                        coord = atom[-1]  # 좌표 (x, y, z)
                        res_coords.append(coord)
                        break

                if coord is None:
                    raise ValueError(f"atom_name '{atom_name}' not found in atom_coords.")

        res_coords = np.array(res_coords)
        transformed_res_coords = random_transform(res_coords)
        ref_pos.extend(transformed_res_coords.tolist())

    ref_space_uid = torch.tensor(ref_space_uid).unsqueeze(0).repeat(B,1)
    ref_element = torch.tensor(ref_element).unsqueeze(0).repeat(B,1,1)
    ref_charge = torch.tensor(ref_charge).unsqueeze(0).repeat(B,1) 
    ref_atom_name_chars = torch.tensor(
        atom_name_chars_encoded(atom_list)
        ).unsqueeze(0).repeat(B,1,1,1) 
    ref_pos = torch.tensor(ref_pos).unsqueeze(0).repeat(B,1,1)
    atom_to_token_idx = torch.tensor(atom_to_token_idx)
    atom_to_tokatom_idx = torch.tensor(atom_to_tokatom_idx)
    return ref_space_uid, ref_element, ref_charge, ref_atom_name_chars, atom_to_token_idx, atom_to_tokatom_idx, ref_pos
