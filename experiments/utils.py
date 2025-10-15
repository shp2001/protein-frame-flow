"""Utility functions for experiments."""
import logging
import torch
import os
import random
import GPUtil
import numpy as np
import pandas as pd
from analysis import utils as au
from data import utils as du
from data import residue_constants as rc 
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from motif_scaffolding import save_motif_segments
from openfold.utils import rigid_utils as ru
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle 

class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [
                int(x) for x in samples_cfg.length_subset
            ]
        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(self._samples_cfg.samples_per_length):
                all_sample_ids.append((length, sample_id))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch


class ScaffoldingDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        self._benchmark_df = pd.read_csv(self._samples_cfg.csv_path)
        if self._samples_cfg.target_subset is not None:
            self._benchmark_df = self._benchmark_df[
                self._benchmark_df.target.isin(self._samples_cfg.target_subset)
            ]
        if len(self._benchmark_df) == 0:
            raise ValueError('No targets found.')
        contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(
            self._benchmark_df)

        num_batch = self._samples_cfg.num_batch
        assert self._samples_cfg.samples_per_target % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_target // num_batch

        all_sample_ids = []
        for row_id in range(len(contigs_by_test_case)):
            target_row = self._benchmark_df.iloc[row_id]
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor([num_batch * sample_id + i for i in range(num_batch)])
                all_sample_ids.append((target_row, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        target_row, sample_id = self._all_sample_ids[idx]
        target = target_row.target
        motif_contig_info = save_motif_segments.load_contig_test_case(target_row)
        motif_segments = [
            torch.tensor(motif_segment, dtype=torch.float64)
            for motif_segment in motif_contig_info['motif_segments']]
        motif_locations  = []
        if isinstance(target_row.length, str):
            lengths = target_row.length.split('-')
            if len(lengths) == 1:
                start_length = lengths[0]
                end_length = lengths[0]
            else:
                start_length, end_length = lengths
            sample_lengths = [int(start_length), int(end_length)+1]
        else:
            sample_lengths = None
        sample_contig, sampled_mask_length, _ = get_sampled_mask(
            motif_contig_info['contig'], sample_lengths)
        motif_locations = save_motif_segments.motif_locations_from_contig(sample_contig[0])
        diffuse_mask = torch.ones(sampled_mask_length)
        trans_1 = torch.zeros(sampled_mask_length, 3)
        rotmats_1 = torch.eye(3)[None].repeat(sampled_mask_length, 1, 1)
        aatype = torch.zeros(sampled_mask_length)
        for (start, end), motif_pos, motif_aatype in zip(motif_locations, motif_segments, motif_contig_info['aatype']):
            diffuse_mask[start:end+1] = 0.0
            motif_rigid = ru.Rigid.from_tensor_7(motif_pos)
            motif_trans = motif_rigid.get_trans()
            motif_rotmats = motif_rigid.get_rots().get_rot_mats()
            trans_1[start:end+1] = motif_trans
            rotmats_1[start:end+1] = motif_rotmats
            aatype[start:end+1] = motif_aatype
        motif_com = torch.sum(trans_1, dim=-2, keepdim=True) / torch.sum(~diffuse_mask.bool())
        trans_1 = diffuse_mask[:, None] * trans_1 + (1 - diffuse_mask[:, None]) * (trans_1 - motif_com)
        return {
            'target': target,
            'sample_id': sample_id,
            'trans_1': trans_1,
            'rotmats_1': rotmats_1,
            'diffuse_mask': diffuse_mask,
            'aatype': aatype,
        }


def get_sampled_mask(contigs, length, rng=None, num_tries=1000000):
    '''
    Parses contig and length argument to sample scaffolds and motifs.

    Taken from rosettafold codebase.
    '''
    length_compatible=False
    count = 0
    while length_compatible is False:
        inpaint_chains=0
        contig_list = contigs.strip().split()
        sampled_mask = []
        sampled_mask_length = 0
        #allow receptor chain to be last in contig string
        if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
            contig_list[-1] = f'{contig_list[-1]},0'
        for con in contig_list:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0'):
                #receptor chain
                sampled_mask.append(con)
            else:
                inpaint_chains += 1
                #chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if '-' in subcon:
                            sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                        else:
                            sampled_mask_length += 1

                    else:
                        if '-' in subcon:
                            if rng is not None:
                                length_inpaint = rng.integers(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            else:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += length_inpaint
                        elif subcon == '0':
                            subcon_out.append('0')
                        else:
                            length_inpaint=int(subcon)
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += int(subcon)
                sampled_mask.append(','.join(subcon_out))
        #check length is compatible 
        if length is not None:
            if sampled_mask_length >= length[0] and sampled_mask_length < length[1]:
                length_compatible = True
        else:
            length_compatible = True
        count+=1
        if count == num_tries: #contig string incompatible with this length
            raise ValueError("Contig string incompatible with --length range")
    return sampled_mask, sampled_mask_length, inpaint_chains


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=True,
    ) 
    eval_dataset = dataset_class(
        dataset_cfg=cfg,
        task=task,
        is_training=False,
    ) 
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit = 8)[:num_device]

def save_conf_repr(input_for_confidence, output_path):
    torch.save(input_for_confidence, output_path)

def save_traj(
        sample: np.ndarray,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        output_dir: str,
        b_factors: np.ndarray,
        diffuse_mask: np.ndarray,
        save_traj_bool,
        aatype = None,
        chain_index = None
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [T, N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        b_factors: [T, N 37]
    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    if b_factors is None:
        b_factor_alt = diffuse_mask
        b_factors = np.tile((b_factor_alt * 100)[:, None], (1, 37))
    
    else:
        b_factors = np.tile((b_factors)[:, None], (1, 37))
    
    sample_path = au.write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=b_factors,
        no_indexing=False,
        aatype=aatype,
        chain_index=chain_index
    )
    if not save_traj_bool:
        return {
            'sample_path': sample_path,
        }
    
    else:
        prot_traj_path = au.write_prot_to_pdb(
            bb_prot_traj,
            prot_traj_path,
            b_factors=b_factors,
            no_indexing=False,
            aatype=aatype,
            chain_index=chain_index
        )
        # x0_traj_path = au.write_prot_to_pdb(
        #     x0_traj,
        #     x0_traj_path,
        #     b_factors=b_factors,
        #     no_indexing=False,
        #     aatype=aatype,
        #     chain_index=chain_index
        # )
        return {
            'sample_path': sample_path,
            'traj_path': prot_traj_path,
            # 'x0_traj_path': x0_traj_path,
        }



def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def dist_map_from_distogram(
        distogram_logit, 
        min_bin=2.0,
        max_bin=22.0,
        do_softmax=True
        ):
    if do_softmax:
        probs = torch.softmax(distogram_logit, dim=-1).detach().cpu().numpy()  # shape: (B, N, N, 32)
    else:
        probs = distogram_logit.detach().cpu().numpy()
        
    num_bins = distogram_logit.shape[-1]  # 32
    bin_edges = np.linspace(min_bin, max_bin, num_bins + 1)  # (33,)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (32,)

    bin_centers = bin_centers.reshape(1, 1, 1, -1)  # shape: (1, 1, 1, 32)
    
    # expected distance 계산
    expected_dmap = np.sum(probs * bin_centers, axis=-1)  # shape: (B, N, N)

    return expected_dmap

def visualize_dist_map(
        dist_map, # (N, N)
        save_path,
        title,
        anchor_residues=None, # 1-dim List
        mark_cdr=False,
        cmap='viridis'
        ):

    plt.figure(figsize=(6, 5))
    plt.imshow(dist_map, cmap=cmap)
    plt.colorbar(label='Expected Distance (Å)')
    plt.title(title)
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')

    if mark_cdr and anchor_residues is not None:
        ax = plt.gca()
        N = dist_map.shape[0]
        for i in range(len(anchor_residues)//2):
            start = anchor_residues[2*i] + 1
            end = anchor_residues[2*i + 1] - 1
            width = end - start + 1

            # 행 강조: 전체 x축에 대해 특정 y 범위에 박스
            rect_row = patches.Rectangle(
                (0, start), N, width,
                linewidth=1.0, edgecolor='red', facecolor='none', linestyle='-', alpha=0.8
            )
            ax.add_patch(rect_row)

    plt.tight_layout()

    # 이미지 저장
    plt.savefig(save_path, dpi=300)  # dpi는 해상도. 필요에 따라 조정 가능


def kabsch_align_full(P, Q, Q_all):
    Pc = P.mean(dim=0, keepdim=True)
    Qc = Q.mean(dim=0, keepdim=True)
    P_centered = P - Pc
    Q_centered = Q - Qc

    H = Q_centered.T @ P_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    Q_all_aligned = (Q_all - Qc) @ R + Pc
    return Q_all_aligned


def calculate_rmsd_info(gt_trans, pred_trans, loop_mask, diffuse_mask):
    """
    gt_trans: (L, 3)
    pred_trans: (L, 3)
    loop_mask: (L,) 0/1 mask
    diffuse_mask: (L,) 0/1 mask (1=antibody, 0=antigen)
    """
    # 1. framework 영역 추출 (diffuse_mask==1 & loop_mask==0)
    framework_idx = torch.nonzero((diffuse_mask==1) & (loop_mask==0), as_tuple=True)[0]
    gt_framework = gt_trans[framework_idx]
    pred_framework = pred_trans[framework_idx]

    # 2. 전체 좌표 정렬
    pred_trans_aligned = kabsch_align_full(gt_framework, pred_framework, pred_trans)

    # 3. loop 구간 추출
    loop_mask_np = loop_mask.cpu().numpy() if isinstance(loop_mask, torch.Tensor) else loop_mask
    loop_indices = []
    in_loop = False
    for i, val in enumerate(loop_mask_np):
        if val == 1 and not in_loop:
            start = i
            in_loop = True
        elif val == 0 and in_loop:
            loop_indices.append((start, i))
            in_loop = False
    if in_loop:
        loop_indices.append((start, len(loop_mask_np)))
    assert len(loop_indices) == 6, f"loop가 6개가 아님: {len(loop_indices)}개 발견됨"

    # 4. loop별 RMSD 계산
    loop_names = ["h1_rms", "h2_rms", "h3_rms", "l1_rms", "l2_rms", "l3_rms"]
    rmsd_info = {}
    for (start, end), name in zip(loop_indices, loop_names):
        gt_loop = gt_trans[start:end, :]
        pred_loop = pred_trans_aligned[start:end, :]
        diff = gt_loop - pred_loop
        rmsd = torch.sqrt(torch.mean(diff.pow(2)))
        rmsd_info[name] = rmsd.item()

    return rmsd_info

def modify_residues_in_cdr(src_pkl, target_dir, cdr_sequence, mutations):
    """
    특정 CDR 서브시퀀스 내에서 여러 residue를 바꾸고
    aatype, atom_mask를 업데이트한 뒤 새로운 pkl 파일로 저장하는 함수.

    Args:
        src_pkl (str): 원본 pkl 파일 경로
        target_dir (str): 수정된 pkl 저장 디렉토리
        cdr_sequence (str): CDR 서브시퀀스 (1-letter 코드)
        mutations (list of tuple): [(rel_pos, new_resname), ...]
            - rel_pos (int): target 내 상대 위치 (0-based)
            - new_resname (str): 교체할 residue (3-letter 코드)
    """
    # 원본 pkl 읽기
    data = du.read_pkl(src_pkl, use_torch=True)

    aatype = data["aatype"].copy()
    atom_mask = data["atom_mask"].copy()   # (L, 37)

    # idx → 1-letter 매핑 준비
    idx_to_resname = {i: r for r, i in rc.resname_to_idx.items()}
    idx_to_aa = {i: rc.restype_3to1.get(res, "X") for i, res in idx_to_resname.items()}

    # 전체 시퀀스 (1-letter)
    seq_full = "".join([idx_to_aa[int(i)] for i in aatype])

    # target 시퀀스 찾기
    start_idx = seq_full.find(cdr_sequence)
    if start_idx == -1:
        raise ValueError("target 시퀀스를 찾을 수 없습니다.")

    # 여러 residue 교체
    for rel_pos, new_resname in mutations:
        replace_idx = start_idx + rel_pos
        # aatype 업데이트
        aatype[replace_idx] = rc.resname_to_idx[new_resname]
        # atom_mask 업데이트
        new_mask = rc.restype_atom37_mask[aatype[replace_idx]]
        atom_mask[replace_idx] = new_mask
        print(f"변경: global_idx={replace_idx}, new_resname={new_resname}")

    # data 업데이트
    data["aatype"] = aatype
    data["atom_mask"] = atom_mask

    # 저장 경로 만들기
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.basename(src_pkl)

    # 파일 이름에 모든 변이 표시
    mut_str = "_".join([f"{pos}{name}" for pos, name in mutations])
    save_path = os.path.join(target_dir, base_name.replace(".pkl", f"_{mut_str}.pkl"))

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ 저장 완료: {save_path}")
    return save_path

def dist_map_from_distogram(
        distogram_logit, 
        min_bin=2.0,
        max_bin=32.0,
        do_softmax=True
        ):
    if do_softmax:
        probs = torch.softmax(distogram_logit, dim=-1).detach().cpu().numpy()  # shape: (B, N, N, 32)
    else:
        probs = distogram_logit.detach().cpu().numpy()
        
    num_bins = distogram_logit.shape[-1]  # 32
    bin_edges = np.linspace(min_bin, max_bin, num_bins + 1)  # (33,)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (32,)

    bin_centers = bin_centers.reshape(1, 1, 1, -1)  # shape: (1, 1, 1, 32)
    
    # expected distance 계산
    expected_dmap = np.sum(probs * bin_centers, axis=-1)  # shape: (B, N, N)

    return expected_dmap

def visualize_dist_map(
        dist_map, # (N, N)
        save_path,
        title,
        cdr_residues=None, # list of tuples, e.g., [(start1, end1), (start2, end2)]
        mark_cdr=False,
        cmap='viridis'
        ):

    plt.figure(figsize=(6, 5))
    
    # vmin과 vmax를 추가하여 colorbar 범위 고정
    plt.imshow(dist_map, cmap=cmap, vmin=0, vmax=20) 
    
    plt.colorbar(label='Expected Distance (Å)')
    plt.title(title)
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')

    if mark_cdr and cdr_residues is not None:
        ax = plt.gca()
        N = dist_map.shape[0]
        for (start, end) in cdr_residues:
            # matplotlib의 Rectangle은 (x, y)에서 시작하여 width, height를 가짐
            # 인덱스가 0부터 시작하고, 칸의 경계를 기준으로 그려야 하므로 -0.5를 해주는 것이 정확함
            start_coord = start - 0.5
            end_coord = end + 0.5
            width = end_coord - start_coord

            # 가로 줄 (Row Highlight)
            rect_row = patches.Rectangle(
                (-0.5, start_coord), N, width,
                linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.9
            )
            # 세로 줄 (Column Highlight)
            rect_col = patches.Rectangle(
                (start_coord, -0.5), width, N,
                linewidth=1.2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.9
            )
            ax.add_patch(rect_row)
            ax.add_patch(rect_col)

    plt.tight_layout()

    # 이미지 저장
    plt.savefig(save_path, dpi=300)
    plt.close() # 메모리 누수 방지를 위해 figure를 닫아주는 것이 좋음