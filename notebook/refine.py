#!/usr/bin/env python3
"""
refine_sidechains_conservative.py

PyRosetta를 이용해 backbone은 완전 고정하고 side-chain도 거의 움직이지 않도록
‘최소 변화(minimal-change)’ 방식으로 부드러운 에너지 정제(refine)를 수행합니다.

특징:
- RotamerTrials 기반의 아주 작은 side-chain sampling
- chi-only minimization (LBFGS) + 매우 작은 움직임
- backbone coordinate constraints
- (옵션) side-chain chi coordinate constraints
"""

import sys
import argparse
import pyrosetta
from pyrosetta import rosetta


# -----------------------------
# 1. PyRosetta 초기화
# -----------------------------
def init_pyrosetta():
    opts = (
        "-mute all "
        "-restore_pre_talaris_2013_behavior false "
        "-detect_disulf true "
    )
    pyrosetta.init(opts)


# -----------------------------
# 2. TaskFactory (repack only)
# -----------------------------
def build_conservative_taskfactory(restrict_to_resnums=None):
    """
    - 디자인 금지
    - 불필요한 side-chain 움직임 최소화 (RotamerTrials 기반)
    - restrict_to_resnums 지정 시 해당 residue만 repack 가능
    """
    tf = rosetta.core.pack.task.TaskFactory()
    tf.push_back(rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(rosetta.core.pack.task.operation.NoRepackDisulfides())

    # 전체적으로 repacking 허용 (하지만 RotamerTrials가 매우 conservative)
    tf.push_back(rosetta.core.pack.task.operation.RestrictToRepacking())

    if restrict_to_resnums:
        # 먼저 모든 residue의 repack을 막고
        tf.push_back(
            rosetta.core.pack.task.operation.OperateOnResidueSubset(
                rosetta.core.pack.task.operation.PreventRepackingRLT(),
                rosetta.core.select.residue_selector.TrueResidueSelector()
            )
        )
        # 지정 residue만 허용
        selector = rosetta.core.select.residue_selector.ResidueIndexSelector()
        selector.set_index(",".join(str(i) for i in restrict_to_resnums))
        tf.push_back(
            rosetta.core.pack.task.operation.OperateOnResidueSubset(
                rosetta.core.pack.task.operation.RestrictToRepackingRLT(),
                selector
            )
        )

    return tf


# -----------------------------
# 3. Backbone + Chi constraint
# -----------------------------
def add_backbone_constraints(pose, scorefxn, k=5.0):
    """
    Backbone 좌표 constraints: 움직임 거의 0
    k=5.0 정도면 χ minimization에서 backbone이 거의 안 움직임.
    """

    constraint_set = rosetta.core.scoring.constraints.ConstraintSet()

    for i in range(1, pose.size() + 1):
        if not pose.residue(i).is_protein():
            continue

        for atom_name in ("N", "CA", "C", "O"):
            if not pose.residue(i).atom_type_set().has_atom(atom_name):
                continue

            atom_id = rosetta.core.id.AtomID(
                pose.residue(i).atom_index(atom_name), i
            )
            xyz = pose.residue(i).xyz(atom_name)

            func = rosetta.core.scoring.func.HarmonicFunc(0.0, k)
            cst = rosetta.core.scoring.constraints.CoordinateConstraint(
                atom_id,
                rosetta.core.id.AtomID(1, 1),  # dummy (origin)
                xyz,
                func
            )
            constraint_set.add_constraint(cst)

    pose.constraint_set(constraint_set)
    scorefxn.set_weight(rosetta.core.scoring.coordinate_constraint, 1.0)


def add_chi_soft_constraints(pose, scorefxn, k=0.1):
    """
    Side-chain chi angle을 원래 위치에서 크게 벗어나지 않도록 아주 약하게 묶어둔다.
    너무 크면 움직임이 거의 안 생김 → 최소한의 refine만 허용.
    """
    for i in range(1, pose.size() + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue

        for chi_idx in range(1, res.nchi() + 1):
            angle0 = res.chi(chi_idx)
            func = rosetta.core.scoring.func.CircularHarmonicFunc(angle0, k)
            cst = rosetta.core.scoring.constraints.DihedralConstraint(
                rosetta.core.id.AtomID(res.atom_index(res.atom_names_for_chi(chi_idx)[0]), i),
                rosetta.core.id.AtomID(res.atom_index(res.atom_names_for_chi(chi_idx)[1]), i),
                rosetta.core.id.AtomID(res.atom_index(res.atom_names_for_chi(chi_idx)[2]), i),
                rosetta.core.id.AtomID(res.atom_index(res.atom_names_for_chi(chi_idx)[3]), i),
                func
            )
            pose.add_constraint(cst)

    scorefxn.set_weight(rosetta.core.scoring.dihedral_constraint, 0.5)


# -----------------------------
# 4. MoveMap (backbone = off)
# -----------------------------
def build_movemap_chi_only():
    mm = rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)    # backbone 고정
    mm.set_chi(True)    # side-chain만
    mm.set_jump(False)
    return mm


# -----------------------------
# 5. Refinement (conservative)
# -----------------------------
def refine_conservative(pose, scorefxn, cycles=2, restrict_to_resnums=None):
    tf = build_conservative_taskfactory(restrict_to_resnums)

    # RotamerTrials = 매우 보수적 side-chain sampling
    rot_trials = rosetta.protocols.minimization_packing.RotamerTrialsMover(scorefxn, tf.create_task_and_apply_taskoperations(pose))


    mm = build_movemap_chi_only()

    # MinMover: 작은 움직임 + LBFGS
    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn)
    min_mover.min_type("lbfgs_armijo_nonmonotone")
    min_mover.tolerance(0.001)
    min_mover.max_iter(50)

    for c in range(cycles):
        print(f"[Cycle {c+1}/{cycles}] RotamerTrials...")
        rot_trials.apply(pose)

        print(f"[Cycle {c+1}/{cycles}] Chi-only minimization...")
        min_mover.apply(pose)

    return pose


# -----------------------------
# 6. main()
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdb")
    parser.add_argument("output_pdb")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--restrict_res", type=str, default=None)
    parser.add_argument("--soft_chi_constraint", action="store_true",
                        help="side-chain이 너무 많이 움직이는 걸 막기 위한 약한 dihedral constraint 적용")
    args = parser.parse_args()

    init_pyrosetta()

    pose = pyrosetta.pose_from_pdb(args.input_pdb)
    scorefxn = rosetta.core.scoring.get_score_function()

    print("Initial score:", scorefxn(pose))

    # 강력 backbone 고정
    add_backbone_constraints(pose, scorefxn)

    # side-chain 움직임도 최소화
    if args.soft_chi_constraint:
        add_chi_soft_constraints(pose, scorefxn)

    restrict = None
    if args.restrict_res:
        restrict = [int(x) for x in args.restrict_res.split(",")]

    refined = refine_conservative(
        pose, scorefxn,
        cycles=args.cycles,
        restrict_to_resnums=restrict
    )

    print("Final score:", scorefxn(refined))
    refined.dump_pdb(args.output_pdb)
    print("Saved:", args.output_pdb)


if __name__ == "__main__":
    main()
