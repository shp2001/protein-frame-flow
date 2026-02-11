import copy
import json
import os.path
import random

import numpy as np
import torch
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from model.model import ProteinMPNN
from prody import *
from sc_utils_for_AbMPNN import Packer, pack_side_chains

torch.set_printoptions(threshold=torch.inf)

class Inference:
    def __init__(self):
        # fmt:off
        # define residue info
        self.alphabet = [
            "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X",
            ]

        self.restype_1to3 = {
            "A": "ALA",
            "R": "ARG",
            "N": "ASN",
            "D": "ASP",
            "C": "CYS",
            "Q": "GLN",
            "E": "GLU",
            "G": "GLY",
            "H": "HIS",
            "I": "ILE",
            "L": "LEU",
            "K": "LYS",
            "M": "MET",
            "F": "PHE",
            "P": "PRO",
            "S": "SER",
            "T": "THR",
            "W": "TRP",
            "Y": "TYR",
            "V": "VAL",
            "X": "UNK",
        }

        self.restype_3to1 = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            "UNK": "X",
        }

        self.restype_int_to_str = {
            0: "A",
            1: "R",
            2: "N",
            3: "D",
            4: "C",
            5: "Q",
            6: "E",
            7: "G",
            8: "H",
            9: "I",
            10: "L",
            11: "K",
            12: "M",
            13: "F",
            14: "P",
            15: "S",
            16: "T",
            17: "W",
            18: "Y",
            19: "V",
            20: "X",
        }

        self.restype_str_to_int = {
            "A": 0,
            "R": 1,
            "N": 2,
            "D": 3,
            "C": 4,
            "Q": 5,
            "E": 6,
            "G": 7,
            "H": 8,
            "I": 9,
            "L": 10,
            "K": 11,
            "M": 12,
            "F": 13,
            "P": 14,
            "S": 15,
            "T": 16,
            "W": 17,
            "Y": 18,
            "V": 19,
            "X": 20,
        }

        self.restype_name_to_atom14_names = {
            "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
            "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
            "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
            "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
            "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
            "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
            "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
            "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
            "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
            "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
            "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
            "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
            "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
            "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
            "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
            "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
            "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
            "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"],
            "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
            "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
            "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        }
        # fmt:on

    def get_aligned_coordinates(self, protein_atoms, CA_dict: dict, atom_name: str):
        """
        protein_atoms: prody atom group
        CA_dict: mapping between chain_residue_idx_icodes and integers
        atom_name: atom to be parsed; e.g. CA
        """
        atom_atoms = protein_atoms.select(f"name {atom_name}")

        if atom_atoms != None:
            atom_coords = atom_atoms.getCoords()
            atom_resnums = atom_atoms.getResnums()
            atom_chain_ids = atom_atoms.getChids()
            atom_icodes = atom_atoms.getIcodes()

        atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
        atom_coords_m = np.zeros([len(CA_dict)], np.int32)
        if atom_atoms != None:
            for i in range(len(atom_resnums)):
                code = (
                    atom_chain_ids[i]
                    + "_"
                    + str(atom_resnums[i])
                    + "_"
                    + atom_icodes[i]
                )
                if code in list(CA_dict):
                    atom_coords_[CA_dict[code], :] = atom_coords[i]
                    atom_coords_m[CA_dict[code]] = 1
        return atom_coords_, atom_coords_m

    def parse_PDB(
        self,
        input_path: str,
        device: str = "cpu",
        chains: list = [],
        parse_all_atoms: bool = False,
        parse_atoms_with_zero_occupancy: bool = False,
    ):
        """
        input_path : path for the input PDB
        device: device for the torch.Tensor
        chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
        parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
        parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
        """
        # fmt:off
        element_list = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mb", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]
        # fmt:on
        element_list = [item.upper() for item in element_list]
        element_dict = dict(zip(element_list, range(1, len(element_list))))

        atom_order = {
            "N": 0,
            "CA": 1,
            "C": 2,
            "CB": 3,
            "O": 4,
            "CG": 5,
            "CG1": 6,
            "CG2": 7,
            "OG": 8,
            "OG1": 9,
            "SG": 10,
            "CD": 11,
            "CD1": 12,
            "CD2": 13,
            "ND1": 14,
            "ND2": 15,
            "OD1": 16,
            "OD2": 17,
            "SD": 18,
            "CE": 19,
            "CE1": 20,
            "CE2": 21,
            "CE3": 22,
            "NE": 23,
            "NE1": 24,
            "NE2": 25,
            "OE1": 26,
            "OE2": 27,
            "CH2": 28,
            "NH1": 29,
            "NH2": 30,
            "OH": 31,
            "CZ": 32,
            "CZ2": 33,
            "CZ3": 34,
            "NZ": 35,
            "OXT": 36,
        }

        if not parse_all_atoms:
            atom_types = ["N", "CA", "C", "O"]
        else:
            atom_types = [
                "N",
                "CA",
                "C",
                "CB",
                "O",
                "CG",
                "CG1",
                "CG2",
                "OG",
                "OG1",
                "SG",
                "CD",
                "CD1",
                "CD2",
                "ND1",
                "ND2",
                "OD1",
                "OD2",
                "SD",
                "CE",
                "CE1",
                "CE2",
                "CE3",
                "NE",
                "NE1",
                "NE2",
                "OE1",
                "OE2",
                "CH2",
                "NH1",
                "NH2",
                "OH",
                "CZ",
                "CZ2",
                "CZ3",
                "NZ",
                "OXT",
            ]

        atoms = parsePDB(input_path)
        if not parse_atoms_with_zero_occupancy:
            atoms = atoms.select("occupancy > 0")
        if chains:
            str_out = ""
            for item in chains:
                str_out += " chain " + item + " or"
            atoms = atoms.select(str_out[1:-3])

        protein_atoms = atoms.select("protein")
        backbone = protein_atoms.select("backbone")
        other_atoms = atoms.select("not protein and not water")
        water_atoms = atoms.select("water")

        CA_atoms = protein_atoms.select("name CA")
        CA_resnums = CA_atoms.getResnums()
        CA_chain_ids = CA_atoms.getChids()
        CA_icodes = CA_atoms.getIcodes()

        CA_dict = {}
        for i in range(len(CA_resnums)):
            code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
            CA_dict[code] = i

        xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
        xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
        for atom_name in atom_types:
            xyz, xyz_m = self.get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
            xyz_37[:, atom_order[atom_name], :] = xyz
            xyz_37_m[:, atom_order[atom_name]] = xyz_m

        N = xyz_37[:, atom_order["N"], :]
        CA = xyz_37[:, atom_order["CA"], :]
        C = xyz_37[:, atom_order["C"], :]
        O = xyz_37[:, atom_order["O"], :]

        N_m = xyz_37_m[:, atom_order["N"]]
        CA_m = xyz_37_m[:, atom_order["CA"]]
        C_m = xyz_37_m[:, atom_order["C"]]
        O_m = xyz_37_m[:, atom_order["O"]]

        mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

        chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
        R_idx = np.array(CA_resnums, dtype=np.int32)
        S = CA_atoms.getResnames()
        S = [
            self.restype_3to1[AA] if AA in list(self.restype_3to1) else "X"
            for AA in list(S)
        ]
        print(f"sequence: {S}")
        S = np.array([self.restype_str_to_int[AA] for AA in list(S)], np.int32)
        X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

        try:
            Y = np.array(other_atoms.getCoords(), dtype=np.float32)
            Y_t = list(other_atoms.getElements())
            Y_t = np.array(
                [
                    element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                    for y_t in Y_t
                ],
                dtype=np.int32,
            )
            Y_m = (Y_t != 1) * (Y_t != 0)

            Y = Y[Y_m, :]
            Y_t = Y_t[Y_m]
            Y_m = Y_m[Y_m]
        except:
            Y = np.zeros([1, 3], np.float32)
            Y_t = np.zeros([1], np.int32)
            Y_m = np.zeros([1], np.int32)

        output_dict = {}
        output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
        output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
        output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
        output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
        output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

        output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
        output_dict["chain_labels"] = torch.tensor(
            chain_labels, device=device, dtype=torch.int32
        )

        output_dict["chain_letters"] = CA_chain_ids

        mask_c = []
        chain_list = list(set(output_dict["chain_letters"]))
        chain_list.sort()
        for chain in chain_list:
            mask_c.append(
                torch.tensor(
                    [chain == item for item in output_dict["chain_letters"]],
                    device=device,
                    dtype=bool,
                )
            )

        output_dict["mask_c"] = mask_c
        output_dict["chain_list"] = chain_list

        output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

        output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
        output_dict["xyz_37_m"] = torch.tensor(
            xyz_37_m, device=device, dtype=torch.int32
        )

        return output_dict, backbone, other_atoms, CA_icodes, water_atoms

    def get_score(self, S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor):
        """
        S : true sequence shape=[batch, length]
        log_probs : predicted sequence shape=[batch, length, 20]
        mask : mask to compute average over the region shape=[batch, length]

        average_loss : averaged categorical cross entropy (CCE) [batch]
        loss_per_resdue : per position CCE [batch, length]
        """
        S_one_hot = torch.nn.functional.one_hot(S, 21)
        S_one_hot = S_one_hot[..., :20]
        loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
        average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
            torch.sum(mask, dim=-1) + 1e-8
        )
        return average_loss, loss_per_residue

    def get_seq_rec(self, S: torch.Tensor, S_pred: torch.Tensor, mask: torch.Tensor):
        """
        S : true sequence shape=[batch, length]
        S_pred : predicted sequence shape=[batch, length]
        mask : mask to compute average over the region shape=[batch, length]

        average : averaged sequence recovery shape=[batch]
        """
        match = S == S_pred
        average = torch.sum(match * mask, dim=-1) / torch.sum(mask, dim=-1)
        return average

    def write_full_PDB(
        self,
        save_path: str,
        X: np.ndarray,
        X_m: np.ndarray,
        b_factors: np.ndarray,
        R_idx: np.ndarray,
        chain_letters: np.ndarray,
        S: np.ndarray,
        other_atoms=None,
        icodes=None,
        force_hetatm=False,
    ):
        """
        save_path : path where the PDB will be written to
        X : protein atom xyz coordinates shape=[length, 14, 3]
        X_m : protein atom mask shape=[length, 14]
        b_factors: shape=[length, 14]
        R_idx: protein residue indices shape=[length]
        chain_letters: protein chain letters shape=[length]
        S : protein amino acid sequence shape=[length]
        other_atoms: other atoms parsed by prody
        icodes: a list of insertion codes for the PDB; e.g. antibody loops
        """

        S_str = [
            self.restype_1to3[AA] for AA in [self.restype_int_to_str[AA] for AA in S]
        ]

        X_list = []
        b_factor_list = []
        atom_name_list = []
        element_name_list = []
        residue_name_list = []
        residue_number_list = []
        chain_id_list = []
        icodes_list = []
        for i, AA in enumerate(S_str):
            sel = X_m[i].astype(np.int32) == 1
            total = np.sum(sel)
            tmp = np.array(self.restype_name_to_atom14_names[AA])[sel]
            X_list.append(X[i][sel])
            b_factor_list.append(b_factors[i][sel])
            atom_name_list.append(tmp)
            element_name_list += [AA[:1] for AA in list(tmp)]
            residue_name_list += total * [AA]
            residue_number_list += total * [R_idx[i]]
            chain_id_list += total * [chain_letters[i]]
            icodes_list += total * [icodes[i]]

        X_stack = np.concatenate(X_list, 0)
        b_factor_stack = np.concatenate(b_factor_list, 0)
        atom_name_stack = np.concatenate(atom_name_list, 0)

        protein = prody.AtomGroup()
        protein.setCoords(X_stack)
        protein.setBetas(b_factor_stack)
        protein.setNames(atom_name_stack)
        protein.setResnames(residue_name_list)
        protein.setElements(element_name_list)
        protein.setOccupancies(np.ones([X_stack.shape[0]]))
        protein.setResnums(residue_number_list)
        protein.setChids(chain_id_list)
        protein.setIcodes(icodes_list)

        if other_atoms:
            other_atoms_g = prody.AtomGroup()
            other_atoms_g.setCoords(other_atoms.getCoords())
            other_atoms_g.setNames(other_atoms.getNames())
            other_atoms_g.setResnames(other_atoms.getResnames())
            other_atoms_g.setElements(other_atoms.getElements())
            other_atoms_g.setOccupancies(other_atoms.getOccupancies())
            other_atoms_g.setResnums(other_atoms.getResnums())
            other_atoms_g.setChids(other_atoms.getChids())
            if force_hetatm:
                other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
            writePDB(save_path, protein + other_atoms_g)
        else:
            writePDB(save_path, protein)

    def featurize(self, input_dict):
        output_dict = {}
        R_idx_list = []
        count = 0
        R_idx_prev = -100000
        for R_idx in list(input_dict["R_idx"]):
            if R_idx_prev == R_idx:
                count += 1
            R_idx_list.append(R_idx + count)
            R_idx_prev = R_idx
        R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
        output_dict["R_idx"] = R_idx_renumbered[None,]
        output_dict["R_idx_original"] = input_dict["R_idx"][None,]
        output_dict["chain_labels"] = input_dict["chain_labels"][None,]
        output_dict["S"] = input_dict["S"][None,]
        output_dict["chain_mask"] = input_dict["chain_mask"][None,]
        output_dict["mask"] = input_dict["mask"][None,]

        output_dict["X"] = input_dict["X"][None,]

        if "xyz_37" in list(input_dict):
            output_dict["xyz_37"] = input_dict["xyz_37"][None,]
            output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

        return output_dict

    def run_inference(self, args, model_param):
        """
        Inference function
        """
        # set seed
        if args.seed:
            seed = args.seed
        else:
            seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"random_seed = {seed}")

        # set device
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

        # set output directories
        base_folder = args.out_folder
        if base_folder[-1] != "/":
            base_folder = base_folder + "/"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        if not os.path.exists(base_folder + "seqs"):
            os.makedirs(base_folder + "seqs", exist_ok=True)
        if not os.path.exists(base_folder + "backbones"):
            os.makedirs(base_folder + "backbones", exist_ok=True)
        if not os.path.exists(base_folder + "packed"):
            os.makedirs(base_folder + "packed", exist_ok=True)
        if args.save_stats:
            if not os.path.exists(base_folder + "stats"):
                os.makedirs(base_folder + "stats", exist_ok=True)

        # load model
        checkpoint_path = args.model
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # fit code to vanilla model
        model_param["num_letters"] = args.out_dim
        model_param["vocab"] = args.out_dim
        if not args.use_refactored_weights:
            model_param["num_letters"] = 21
            model_param["vocab"] = 21
            # fmt:off
            self.restype_str_to_int = { "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19, "X": 20,}
            self.restype_int_to_str = { 0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I", 8: "K", 9: "L", 10: "M", 11: "N", 12: "P", 13: "Q", 14: "R", 15: "S", 16: "T", 17: "V", 18: "W", 19: "Y", 20: "X",}
            # fmt:on
        model = ProteinMPNN(**model_param)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        # load model for sc packing option
        if args.pack_side_chains:
            model_sc = Packer(
                node_features=128,
                edge_features=128,
                num_positional_embeddings=16,
                num_chain_embeddings=16,
                num_rbf=16,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                atom_context_num=16,
                lower_bound=0.0,
                upper_bound=20.0,
                top_k=32,
                dropout=0.0,
                augment_eps=0.0,
                atom37_order=False,
                device=device,
                num_mix=3,
            )

            checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device)
            model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
            model_sc.to(device)
            model_sc.eval()

        ########################################
        # get inputs from args                 #
        ########################################
        # get input PDB paths
        if args.pdb_path_multi:
            with open(args.pdb_path_multi, "r") as fh:
                pdb_paths = list(json.load(fh))
        else:
            pdb_paths = [args.pdb_path]

        if args.multi_state:
            if args.pdb_path_alt_multi:
                with open(args.pdb_path_alt_multi, "r") as fh:
                    pdb_paths_alt = iter(list(json.load(fh)))
            else:
                pdb_paths_alt = iter([args.pdb_path_alt])

        # get fixed_residues
        if args.fixed_residues_multi:
            with open(args.fixed_residues_multi, "r") as fh:
                fixed_residues_multi = json.load(fh)
            for key, value in fixed_residues_multi.items():
                if isinstance(value, str):
                    fixed_residues_multi[key] = value.split()
        else:
            fixed_residues = args.fixed_residues.split()
            fixed_residues_multi = {}
            for pdb in pdb_paths:
                fixed_residues_multi[pdb] = fixed_residues

        # get redesigned_residues
        if args.redesigned_residues_multi:
            with open(args.redesigned_residues_multi, "r") as fh:
                redesigned_residues_multi = json.load(fh)
            for key, value in redesigned_residues_multi.items():
                if isinstance(value, str):
                    redesigned_residues_multi[key] = value.split()
        else:
            redesigned_residues = args.redesigned_residues.split()
            redesigned_residues_multi = {}
            for pdb in pdb_paths:
                redesigned_residues_multi[pdb] = redesigned_residues

        # get bias_AA & bias_AA_per_residue
        bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
        if args.bias_AA:
            tmp = [item.split(":") for item in args.bias_AA.split(",")]
            a1 = [b[0] for b in tmp]
            a2 = [float(b[1]) for b in tmp]
            for i, AA in enumerate(a1):
                bias_AA[self.restype_str_to_int[AA]] = a2[i]

        if args.bias_AA_per_residue_multi:
            with open(args.bias_AA_per_residue_multi, "r") as fh:
                bias_AA_per_residue_multi = json.load(
                    fh
                )  # {"pdb_path" : {"A12": {"G": 1.1}}}
        else:
            if args.bias_AA_per_residue:
                with open(args.bias_AA_per_residue, "r") as fh:
                    bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
                bias_AA_per_residue_multi = {}
                for pdb in pdb_paths:
                    bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

        # get omit_AA & omit_AA_per_residue
        if args.omit_AA_per_residue_multi:
            with open(args.omit_AA_per_residue_multi, "r") as fh:
                omit_AA_per_residue_multi = json.load(
                    fh
                )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
        else:
            if args.omit_AA_per_residue:
                with open(args.omit_AA_per_residue, "r") as fh:
                    omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
                omit_AA_per_residue_multi = {}
                for pdb in pdb_paths:
                    omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
        omit_AA_list = args.omit_AA
        omit_AA = torch.tensor(
            np.array([AA in omit_AA_list for AA in self.alphabet]).astype(np.float32),
            device=device,
        )

        # get parse_these_chains_only
        if len(args.parse_these_chains_only) != 0:
            parse_these_chains_only_list = args.parse_these_chains_only.split(",")
        else:
            parse_these_chains_only_list = []

        ########################################
        # loop over PDB paths                  #
        ########################################
        for pdb in pdb_paths:
            if args.verbose:
                print("Designing protein from this path:", pdb)
            fixed_residues = fixed_residues_multi[pdb]
            redesigned_residues = redesigned_residues_multi[pdb]
            parse_all_atoms_flag = args.pack_side_chains and not args.repack_everything

            # parse PDB
            protein_dict, backbone, other_atoms, icodes, _ = self.parse_PDB(
                pdb,
                device=device,
                chains=parse_these_chains_only_list,
                parse_all_atoms=parse_all_atoms_flag,
                parse_atoms_with_zero_occupancy=bool(
                    args.parse_atoms_with_zero_occupancy
                ),
            )

            if args.multi_state:
                pdb_alt = next(pdb_paths_alt)
                protein_dict_alt, backbone_alt, other_atoms_alt, icodes_alt, _ = (
                    self.parse_PDB(
                        pdb_alt,
                        device=device,
                        chains=parse_these_chains_only_list,
                        parse_all_atoms=parse_all_atoms_flag,
                        parse_atoms_with_zero_occupancy=bool(
                            args.parse_atoms_with_zero_occupancy
                        ),
                    )
                )

            # make chain_letter + residue_idx + insertion_code mapping to integers
            R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
            chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
            encoded_residues = []
            for i, R_idx_item in enumerate(R_idx_list):
                tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
                encoded_residues.append(tmp)
            encoded_residue_dict = dict(
                zip(encoded_residues, range(len(encoded_residues)))
            )
            encoded_residue_dict_rev = dict(
                zip(list(range(len(encoded_residues))), encoded_residues)
            )

            if args.multi_state:
                R_idx_list_alt = list(protein_dict_alt["R_idx"].cpu().numpy())
                chain_letters_list_alt = list(protein_dict_alt["chain_letters"])
                encoded_residues_alt = []
                for i, R_idx_item in enumerate(R_idx_list_alt):
                    tmp = (
                        str(chain_letters_list_alt[i]) + str(R_idx_item) + icodes_alt[i]
                    )
                    encoded_residues_alt.append(tmp)
                encoded_residue_dict_rev_alt = dict(
                    zip(list(range(len(encoded_residues_alt))), encoded_residues_alt)
                )

            # convert bias_AA_per_residue to tensor
            bias_AA_per_residue = torch.zeros(
                [len(encoded_residues), 21], device=device, dtype=torch.float32
            )
            if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
                bias_dict = bias_AA_per_residue_multi[pdb]
                for residue_name, v1 in bias_dict.items():
                    if residue_name in encoded_residues:
                        i1 = encoded_residue_dict[residue_name]
                        for amino_acid, v2 in v1.items():
                            if amino_acid in self.alphabet:
                                j1 = self.restype_str_to_int[amino_acid]
                                bias_AA_per_residue[i1, j1] = v2

            # convert omit_AA_per_residue to tensor
            omit_AA_per_residue = torch.zeros(
                [len(encoded_residues), 21], device=device, dtype=torch.float32
            )
            if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
                omit_dict = omit_AA_per_residue_multi[pdb]
                for residue_name, v1 in omit_dict.items():
                    if residue_name in encoded_residues:
                        i1 = encoded_residue_dict[residue_name]
                        for amino_acid in v1:
                            if amino_acid in self.alphabet:
                                j1 = self.restype_str_to_int[amino_acid]
                                omit_AA_per_residue[i1, j1] = 1.0

            # convert fixed_positions to tensor
            fixed_positions = torch.tensor(
                [int(item not in fixed_residues) for item in encoded_residues],
                device=device,
            )

            # convert redesigned_positions to tensor
            redesigned_positions = torch.tensor(
                [int(item not in redesigned_residues) for item in encoded_residues],
                device=device,
            )

            # get input chains_to_design list, or just design all chains
            if len(args.chains_to_design) != 0:
                chains_to_design_list = args.chains_to_design.split(",")
            else:
                chains_to_design_list = protein_dict["chain_letters"]

            # create chain_mask and initialize with chains_to_design_list
            chain_mask = torch.tensor(
                np.array(
                    [
                        item in chains_to_design_list
                        for item in protein_dict["chain_letters"]
                    ],
                    dtype=np.int32,
                ),
                device=device,
            )

            # update chain_mask to notify which residues are fixed (0) and which need to be designed (1), and add to protein_dict
            if args.redesigned_residues is not None and args.redesigned_residues.strip() == "": ## NOTE : added logic (redesigned == "" -> no design)
                protein_dict["chain_mask"] = torch.zeros_like(chain_mask)
            elif redesigned_residues:
                protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
                if fixed_residues:
                    print(
                        "fixed_residues option is ignored because of redesigned_residues option"
                    )
            elif fixed_residues:
                protein_dict["chain_mask"] = chain_mask * fixed_positions
            else:
                protein_dict["chain_mask"] = chain_mask

            if args.multi_state:
                protein_dict_alt["chain_mask"] = protein_dict["chain_mask"]

            if args.verbose:
                PDB_residues_to_be_redesigned = [
                    encoded_residue_dict_rev[item]
                    for item in range(protein_dict["chain_mask"].shape[0])
                    if protein_dict["chain_mask"][item] == 1
                ]
                PDB_residues_to_be_fixed = [
                    encoded_residue_dict_rev[item]
                    for item in range(protein_dict["chain_mask"].shape[0])
                    if protein_dict["chain_mask"][item] == 0
                ]
                print(
                    "These residues will be redesigned: ", PDB_residues_to_be_redesigned
                )
                print("These residues will be fixed: ", PDB_residues_to_be_fixed)

            # specify which residues are linked
            if args.symmetry_residues:
                symmetry_residues_list_of_lists = [
                    x.split(",") for x in args.symmetry_residues.split("|")
                ]
                remapped_symmetry_residues = []
                for t_list in symmetry_residues_list_of_lists:
                    tmp_list = []
                    for t in t_list:
                        tmp_list.append(encoded_residue_dict[t])
                    remapped_symmetry_residues.append(tmp_list)
            else:
                remapped_symmetry_residues = [[]]

            # specify linking weights
            if args.symmetry_weights:
                symmetry_weights = [
                    [float(item) for item in x.split(",")]
                    for x in args.symmetry_weights.split("|")
                ]
            else:
                symmetry_weights = [[]]

            if args.homo_oligomer or args.symmetry_chains:
                if args.symmetry_residues:
                    print(
                        "Symmetry residues are ignored because other symmetry option is selected."
                    )
                if args.homo_oligomer and args.symmetry_chains:
                    print(
                        "symmetry_chains option is ignored because homo_oligomer option is selected."
                    )
                if args.verbose:
                    if args.homo_oligomer:
                        print("Designing HOMO-OLIGOMER")
                    elif args.symmetry_chains:
                        print(
                            "Designing with chain symmetry:",
                            args.symmetry_chains,
                        )

                remapped_symmetry_residues = []
                symmetry_weights = []
                if args.homo_oligomer:
                    chain_letters_set_list = [protein_dict["chain_list"]]
                elif args.symmetry_chains:
                    chain_letters_set_list = [
                        chains.split(",") for chains in args.symmetry_chains.split("|")
                    ]
                for chain_letters_set in chain_letters_set_list:
                    reference_chain = chain_letters_set[0]
                    len_chain_letter = len(reference_chain)
                    residue_indices = [
                        item[len_chain_letter:]
                        for item in encoded_residues
                        if item[:len_chain_letter] == reference_chain
                    ]
                    for res in residue_indices:
                        tmp_list = []
                        for chain in chain_letters_set:
                            name = chain + res
                            if name in encoded_residue_dict:
                                tmp_list.append(encoded_residue_dict[name])
                        tmp_w_list = [1 / len(tmp_list) for _ in tmp_list]
                        remapped_symmetry_residues.append(tmp_list)
                        symmetry_weights.append(tmp_w_list)

            # set other atom bfactors to 0.0
            if other_atoms:
                other_bfactors = other_atoms.getBetas()
                other_atoms.setBetas(other_bfactors * 0.0)

            # adjust input PDB name by dropping .pdb if it does exist
            name = pdb[pdb.rfind("/") + 1 :]
            if name[-4:] == ".pdb":
                name = name[:-4]

            ########################################
            # featurize inputs and run inference   #
            ########################################
            with torch.no_grad():
                # run featurize to remap R_idx and add batch dimension
                if args.verbose:
                    if "Y" in list(protein_dict):
                        atom_coords = protein_dict["Y"].cpu().numpy()
                        atom_types = list(protein_dict["Y_t"].cpu().numpy())
                        atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                        number_of_atoms_parsed = np.sum(atom_mask)
                    else:
                        print("No ligand atoms parsed")
                        number_of_atoms_parsed = 0
                        atom_types = ""
                        atom_coords = []

                    if number_of_atoms_parsed == 0:
                        print("No ligand atoms parsed")

                feature_dict = self.featurize(protein_dict)
                feature_dict["batch_size"] = args.batch_size
                _, L, _, _ = feature_dict["X"].shape
                # add additional keys to the feature dictionary
                feature_dict["temperature"] = args.temperature
                feature_dict["bias"] = (
                    (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                    + bias_AA_per_residue[None]
                    - 1e8 * omit_AA_per_residue[None]
                )
                feature_dict["symmetry_residues"] = remapped_symmetry_residues
                feature_dict["symmetry_weights"] = symmetry_weights
                feature_dict["decoding_type"] = args.decoding_type
                feature_dict["N_patch"] = args.N_patch
                feature_dict["use_refactored_weights"] = args.use_refactored_weights

                if not args.score: # NOTE : logic added - in sampling mode, if no design positions (redesigned_residues == ""), skip sampling & WT seq sc packing+output
                    design_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                    num_design_positions = int(design_mask.sum().item())
                    if num_design_positions == 0:
                        if args.verbose:
                            print(
                                f"[{name}] No residues marked for redesign "
                                "(chain_mask all zeros). "
                                "Running native-sequence packing only."
                            )

                        native_seq = "".join(
                            [
                                self.restype_int_to_str[AA]
                                for AA in feature_dict["S"][0].cpu().numpy()
                            ]
                        )
                        seq_np = np.array(list(native_seq))
                        seq_out_str = []
                        for mask in protein_dict["mask_c"]:
                            seq_out_str += list(seq_np[mask.cpu().numpy()])
                            seq_out_str += [args.fasta_seq_separation]
                        seq_out_str = "".join(seq_out_str)[:-1]

                        output_fasta = (
                            base_folder + "/seqs/" + name + args.file_ending + ".fa"
                        )
                        output_backbones = (
                            base_folder + "/backbones/"
                        )
                        output_packed = base_folder + "/packed/"

                        num_res_native = int(feature_dict["mask"].sum().item())
                        with open(output_fasta, "w") as f:
                            f.write(
                                ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                                    name,
                                    args.temperature,
                                    seed,
                                    num_res_native,
                                    0,
                                    1,
                                    0,
                                    checkpoint_path,
                                    seq_out_str,
                                )
                            )

                        seq_prody = np.array(
                            [self.restype_1to3[AA] for AA in list(native_seq)]
                        )
                        residues = backbone.getHierView().iterResidues()
                        for residue, resname in zip(residues, seq_prody):
                            residue.setResname(resname)
                            residue.setBetas(
                                np.ones_like(residue.getBetas(), dtype=np.float32)
                            )

                        with open(
                            output_backbones
                            + name
                            + "_native"
                            + args.file_ending
                            + ".pdb",
                            "w",
                        ) as f_pdb:
                            writePDBStream(
                                f_pdb,
                                backbone + other_atoms if other_atoms else backbone,
                            )

                        if args.pack_side_chains:
                            if args.verbose:
                                print("Packing side chains for native sequence...")
                            B_sc = 1
                            feature_dict_ = self.featurize(protein_dict)
                            sc_feature_dict = copy.deepcopy(feature_dict_)
                            for k, v in sc_feature_dict.items():
                                if k != "S":
                                    try:
                                        num_dim = len(v.shape)
                                        if num_dim == 2:
                                            sc_feature_dict[k] = v.repeat(B_sc, 1)
                                        elif num_dim == 3:
                                            sc_feature_dict[k] = v.repeat(B_sc, 1, 1)
                                        elif num_dim == 4:
                                            sc_feature_dict[k] = v.repeat(
                                                B_sc, 1, 1, 1
                                            )
                                        elif num_dim == 5:
                                            sc_feature_dict[k] = v.repeat(
                                                B_sc, 1, 1, 1, 1
                                            )
                                    except:
                                        pass
                            
                            S_native = feature_dict_["S"] # native seq
                            if S_native.dim() == 1:
                                # [L] -> [1, L]
                                S_native = S_native.unsqueeze(0)
                            S_native = S_native.long()
                            sc_feature_dict["S"] = S_native

                            for c_pack in range(args.number_of_packs_per_design):
                                sc_dict = pack_side_chains(
                                    sc_feature_dict,
                                    model_sc,
                                    args.sc_num_denoising_steps,
                                    args.sc_num_samples,
                                    args.repack_everything,
                                )
                                X = sc_dict["X"][0].cpu().numpy()
                                X_m = sc_dict["X_m"][0].cpu().numpy()
                                b_f = sc_dict["b_factors"][0].cpu().numpy()

                                self.write_full_PDB(
                                    output_packed
                                    + name
                                    + args.packed_suffix
                                    + "_native_"
                                    + str(c_pack + 1)
                                    + args.file_ending
                                    + ".pdb",
                                    X,
                                    X_m,
                                    b_f,
                                    feature_dict["R_idx_original"][0].cpu().numpy(),
                                    protein_dict["chain_letters"],
                                    feature_dict["S"][0].cpu().numpy(),
                                    other_atoms=other_atoms,
                                    icodes=icodes,
                                    force_hetatm=args.force_hetatm,
                                )
                        continue
                
                feature_dict_alt = None
                if args.multi_state:
                    feature_dict_alt = self.featurize(protein_dict_alt)
                    feature_dict_alt["multi_state_weight"] = args.multi_state_weight

                if args.score:
                    # initialize scoring parameters
                    logits_list = []
                    probs_list = []
                    log_probs_list = []
                    decoding_order_list = []
                else:
                    # initialze prediction parameters
                    sampling_probs_list = []
                    log_probs_list = []
                    decoding_order_list = []
                    S_list = []
                    loss_list = []
                    loss_per_residue_list = []
                    loss_XY_list = []
                for batch_i in range(args.number_of_batches):
                    feature_dict["randn"] = torch.randn(
                        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                        device=device,
                    )
                    
                    # if batch_i > 0: # for artificial sync w/ fullmoon original ver
                    #     [
                    #         torch.rand(1, device=device)
                    #         for _ in range((L + 1) * args.batch_size - 1)
                    #     ]

                    if args.score:
                        if args.autoregressive_score:
                            score_dict = model.score(
                                feature_dict, use_sequence=args.use_sequence
                            )
                        elif args.single_aa_score:
                            score_dict = model.single_aa_score(
                                feature_dict, use_sequence=args.use_sequence
                            )
                        else:
                            print(
                                "Set either autoregressive_score or single_aa_score to True"
                            )
                            return
                        logits_list.append(score_dict["logits"])
                        log_probs_list.append(score_dict["log_probs"])
                        probs_list.append(torch.exp(score_dict["log_probs"]))
                        decoding_order_list.append(score_dict["decoding_order"])
                    else:
                        # predict protein sequence
                        output_dict = model.sample(feature_dict, feature_dict_alt)
                        print(
                            f"decoding_type and N_patch : {feature_dict['decoding_type']}, {feature_dict['N_patch']}"
                        )

                        # compute confidence scores
                        loss, loss_per_residue = self.get_score(
                            output_dict["S"],
                            output_dict["log_probs"],
                            feature_dict["mask"] * feature_dict["chain_mask"],
                        )

                        combined_mask = (
                            feature_dict["mask"] * feature_dict["chain_mask"]
                        )
                        loss_XY, _ = self.get_score(
                            output_dict["S"], output_dict["log_probs"], combined_mask
                        )

                        # get output values and update output lists
                        S_list.append(output_dict["S"])
                        log_probs_list.append(output_dict["log_probs"])
                        sampling_probs_list.append(output_dict["sampling_probs"])
                        decoding_order_list.append(output_dict["decoding_order"])
                        loss_list.append(loss)
                        loss_per_residue_list.append(loss_per_residue)
                        loss_XY_list.append(loss_XY)

                if args.score:
                    # Write scoring output file.
                    log_probs_stack = torch.cat(log_probs_list, 0)
                    logits_stack = torch.cat(logits_list, 0)
                    probs_stack = torch.cat(probs_list, 0)
                    decoding_order_stack = torch.cat(decoding_order_list, 0)

                    output_stats_path = base_folder + name + args.file_ending + ".pt"
                    out_dict = {}
                    out_dict["logits"] = logits_stack.cpu().numpy()
                    out_dict["probs"] = probs_stack.cpu().numpy()
                    out_dict["log_probs"] = log_probs_stack.cpu().numpy()
                    out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
                    out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
                    out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
                    out_dict["chain_mask"] = (
                        feature_dict["chain_mask"][0].cpu().numpy()
                    )  # this affects decoding order
                    out_dict["seed"] = seed
                    out_dict["alphabet"] = self.alphabet
                    out_dict["residue_names"] = encoded_residue_dict_rev

                    mean_probs = np.mean(out_dict["probs"], 0)
                    std_probs = np.std(out_dict["probs"], 0)
                    sequence = [
                        self.restype_int_to_str[AA]
                        for AA in out_dict["native_sequence"]
                    ]
                    mean_dict = {}
                    std_dict = {}
                    for residue in range(L):
                        mean_dict_ = dict(zip(self.alphabet, mean_probs[residue]))
                        mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
                        std_dict_ = dict(zip(self.alphabet, std_probs[residue]))
                        std_dict[encoded_residue_dict_rev[residue]] = std_dict_

                    out_dict["sequence"] = sequence
                    out_dict["mean_of_probs"] = mean_dict
                    out_dict["std_of_probs"] = std_dict
                    out_dict["model"] = args.model
                    torch.save(out_dict, output_stats_path, pickle_protocol=5)
                    continue

                # gather output lists to single tensor
                S_stack = torch.cat(S_list, 0)
                log_probs_stack = torch.cat(log_probs_list, 0)
                sampling_probs_stack = torch.cat(sampling_probs_list, 0)
                decoding_order_stack = torch.cat(decoding_order_list, 0)
                loss_stack = torch.cat(loss_list, 0)
                loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
                loss_XY_stack = torch.cat(loss_XY_list, 0)
                rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
                rec_stack = self.get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

                # convert to alphabet sequence
                native_seq = "".join(
                    [
                        self.restype_int_to_str[AA]
                        for AA in feature_dict["S"][0].cpu().numpy()
                    ]
                )
                seq_np = np.array(list(native_seq))
                seq_out_str = []

                # seperate output sequence by chain
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [args.fasta_seq_separation]
                seq_out_str = "".join(seq_out_str)[:-1]

                # directory and file to save outputs
                output_fasta = base_folder + "/seqs/" + name + args.file_ending + ".fa"
                output_backbones = base_folder + "/backbones/"
                output_packed = base_folder + "/packed/"
                output_stats_path = (
                    base_folder + "stats/" + name + args.file_ending + ".pt"
                )

                # save outputs as dictionary
                out_dict = {}
                out_dict["generated_sequences"] = S_stack.cpu()
                out_dict["sampling_probs"] = sampling_probs_stack.cpu()
                out_dict["log_probs"] = log_probs_stack.cpu()
                out_dict["decoding_order"] = decoding_order_stack.cpu()
                print(decoding_order_stack)
                out_dict["native_sequence"] = feature_dict["S"][0].cpu()
                out_dict["mask"] = feature_dict["mask"][0].cpu()
                out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
                out_dict["seed"] = seed
                out_dict["temperature"] = args.temperature
                if args.multi_state:
                    out_dict["mask_alt"] = feature_dict_alt["mask"][0].cpu()
                if args.save_stats:
                    torch.save(out_dict, output_stats_path)

                ########################################
                # pack side chains                     #
                ########################################
                if args.pack_side_chains:
                    if args.verbose:
                        print("Packing side chains...")
                    feature_dict_ = self.featurize(protein_dict)
                    sc_feature_dict = copy.deepcopy(feature_dict_)
                    B = args.batch_size
                    for k, v in sc_feature_dict.items():
                        if k != "S":
                            try:
                                num_dim = len(v.shape)
                                if num_dim == 2:
                                    sc_feature_dict[k] = v.repeat(B, 1)
                                elif num_dim == 3:
                                    sc_feature_dict[k] = v.repeat(B, 1, 1)
                                elif num_dim == 4:
                                    sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                                elif num_dim == 5:
                                    sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                            except:
                                pass
                    X_stack_list = []
                    X_m_stack_list = []
                    b_factor_stack_list = []
                    for _ in range(args.number_of_packs_per_design):
                        X_list = []
                        X_m_list = []
                        b_factor_list = []
                        for c in range(args.number_of_batches):
                            sc_feature_dict["S"] = S_list[c]
                            sc_dict = pack_side_chains(
                                sc_feature_dict,
                                model_sc,
                                args.sc_num_denoising_steps,
                                args.sc_num_samples,
                                args.repack_everything,
                            )
                            X_list.append(sc_dict["X"])
                            X_m_list.append(sc_dict["X_m"])
                            b_factor_list.append(sc_dict["b_factors"])

                        X_stack = torch.cat(X_list, 0)
                        X_m_stack = torch.cat(X_m_list, 0)
                        b_factor_stack = torch.cat(b_factor_list, 0)

                        X_stack_list.append(X_stack)
                        X_m_stack_list.append(X_m_stack)
                        b_factor_stack_list.append(b_factor_stack)

                ########################################
                # write output files                   #
                ########################################
                removed_residues = [
                    encoded_residue_dict_rev[res_idx]
                    for res_idx, mask_value in enumerate(out_dict["mask"])
                    if not mask_value
                ]
                if removed_residues:
                    print(
                        "These residues were invalid and removed:",
                        removed_residues,
                    )
                if args.multi_state:
                    removed_residues_alt = [
                        encoded_residue_dict_rev_alt[res_idx]
                        for res_idx, mask_value in enumerate(out_dict["mask_alt"])
                        if not mask_value
                    ]
                with open(output_fasta, "w") as f:
                    f.write(
                        ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                            name,
                            args.temperature,
                            seed,
                            torch.sum(rec_mask).cpu().numpy(),
                            torch.sum(combined_mask[:1]).cpu().numpy(),
                            args.batch_size,
                            args.number_of_batches,
                            checkpoint_path,
                            seq_out_str,
                        )
                    )
                    for ix in range(S_stack.shape[0]):
                        ix_suffix = ix
                        if not args.zero_indexed:
                            ix_suffix += 1
                        seq_rec_print = np.format_float_positional(
                            rec_stack[ix].cpu().numpy(), unique=False, precision=4
                        )
                        loss_np = np.format_float_positional(
                            np.exp(-loss_stack[ix].cpu().numpy()),
                            unique=False,
                            precision=4,
                        )
                        loss_XY_np = np.format_float_positional(
                            np.exp(-loss_XY_stack[ix].cpu().numpy()),
                            unique=False,
                            precision=4,
                        )
                        seq = "".join(
                            [
                                self.restype_int_to_str[AA]
                                for AA in S_stack[ix].cpu().numpy()
                            ]
                        )

                        # write new sequences into PDB with backbone coordinates
                        seq_prody = np.array(
                            [self.restype_1to3[AA] for AA in list(seq)]
                        )  # (L)
                        bfactor_prody = loss_per_residue_stack[ix].cpu().numpy()  # (L)
                        residues = backbone.getHierView().iterResidues()
                        for residue, resname, bfactor in zip(
                            residues, seq_prody, bfactor_prody
                        ):
                            residue.setResname(resname)
                            residue.setBetas(
                                np.exp(-bfactor) * (bfactor > 0.01).astype(np.float32)
                            )

                        # mask out residues
                        if ix == 0:
                            residues = backbone.getHierView().iterResidues()
                            residue_indices = [res.getResindex() for res in residues]
                            selection = " or ".join(
                                [
                                    f"resindex {res_idx}"
                                    for res_idx, mask_value in zip(
                                        residue_indices, out_dict["mask"]
                                    )
                                    if mask_value
                                ]
                            )
                        backbone_ = backbone.select(selection)

                        if args.multi_state:
                            residues_alt = backbone_alt.getHierView().iterResidues()
                            for residue, resname, bfactor in zip(
                                residues_alt, seq_prody, bfactor_prody
                            ):
                                residue.setResname(resname)
                                residue.setBetas(
                                    np.exp(-bfactor)
                                    * (bfactor > 0.01).astype(np.float32)
                                )

                            if ix == 0:
                                residues_alt = backbone_alt.getHierView().iterResidues()
                                residue_indices_alt = [
                                    res.getResindex() for res in residues_alt
                                ]
                                selection_alt = " or ".join(
                                    [
                                        f"resindex {res_idx}"
                                        for res_idx, mask_value in zip(
                                            residue_indices_alt, out_dict["mask_alt"]
                                        )
                                        if mask_value
                                    ]
                                )
                            backbone_alt_ = backbone_alt.select(selection_alt)

                        with open(
                            output_backbones
                            + name
                            + "_"
                            + str(ix_suffix)
                            + args.file_ending
                            + ".pdb",
                            "w",
                        ) as f_pdb:
                            remarks = (
                                [
                                    "REMARK These residues were invalid and removed.\n",
                                    f"REMARK {' '.join(removed_residues)}\n",
                                ]
                                if removed_residues
                                else []
                            )
                            for line in remarks:
                                f_pdb.write(line)
                            writePDBStream(
                                f_pdb,
                                backbone_ + other_atoms if other_atoms else backbone_,
                            )
                        if args.multi_state:
                            with open(
                                output_backbones
                                + name
                                + "_alt"
                                + "_"
                                + str(ix_suffix)
                                + args.file_ending
                                + ".pdb",
                                "w",
                            ) as f_pdb:
                                remarks = (
                                    [
                                        "REMARK These residues were invalid and removed.\n",
                                        f"REMARK {' '.join(removed_residues_alt)}\n",
                                    ]
                                    if removed_residues_alt
                                    else []
                                )
                                for line in remarks:
                                    f_pdb.write(line)
                                writePDBStream(
                                    f_pdb,
                                    (
                                        backbone_alt_ + other_atoms_alt
                                        if other_atoms_alt
                                        else backbone_alt_
                                    ),
                                )

                        # write full PDB files (if pack_side_chains)
                        if args.pack_side_chains:
                            for c_pack in range(args.number_of_packs_per_design):
                                X_stack = X_stack_list[c_pack]
                                X_m_stack = X_m_stack_list[c_pack]
                                b_factor_stack = b_factor_stack_list[c_pack]
                                self.write_full_PDB(
                                    output_packed
                                    + name
                                    + args.packed_suffix
                                    + "_"
                                    + str(ix_suffix)
                                    + "_"
                                    + str(c_pack + 1)
                                    + args.file_ending
                                    + ".pdb",
                                    X_stack[ix].cpu().numpy(),
                                    X_m_stack[ix].cpu().numpy(),
                                    b_factor_stack[ix].cpu().numpy(),
                                    feature_dict["R_idx_original"][0].cpu().numpy(),
                                    protein_dict["chain_letters"],
                                    S_stack[ix].cpu().numpy(),
                                    other_atoms=other_atoms,
                                    icodes=icodes,
                                    force_hetatm=args.force_hetatm,
                                )

                        # write fasta lines
                        seq_np = np.array(list(seq))
                        seq_out_str = []
                        for mask in protein_dict["mask_c"]:
                            seq_out_str += list(seq_np[mask.cpu().numpy()])
                            seq_out_str += [args.fasta_seq_separation]
                        seq_out_str = "".join(seq_out_str)[:-1]
                        if ix == S_stack.shape[0] - 1:
                            # final 2 lines
                            f.write(
                                ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
                                    name,
                                    ix_suffix,
                                    args.temperature,
                                    seed,
                                    loss_np,
                                    loss_XY_np,
                                    seq_rec_print,
                                    seq_out_str,
                                )
                            )
                        else:
                            f.write(
                                ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
                                    name,
                                    ix_suffix,
                                    args.temperature,
                                    seed,
                                    loss_np,
                                    loss_XY_np,
                                    seq_rec_print,
                                    seq_out_str,
                                )
                            )


if __name__ == "__main__":
    from arguments import get_args_inference

    args, model_param = get_args_inference()

    Inference().run_inference(args, model_param)
