import os 
import shutil 
import wandb  
import pandas as pd 
import json 
import random 

import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader

from models.flow_model import ConfidenceModel
from models.loss import compute_prmsd_loss, lddt_loss

import data.utils as du 

from experiments.inference_se3_flows import EvalRunner

from omegaconf import OmegaConf
from datetime import datetime
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))


def sample_csv(
    input_csv_path: str,
    cluster_path: str,
    output_csv_path: str,
    epoch: int,
    general_sample_size: int = 1000,
    random_state: int = 42
):
    """
    클러스터에서 각 클러스터별 하나의 pdb_name을 선택하고, general mode 중 일부를 샘플링하여 새 CSV 저장.

    Parameters:
        input_csv_path (str): 원본 CSV 경로
        cluster_path (str): 클러스터 딕셔너리(pickle 등) 경로
        output_csv_path (str): 저장할 CSV 경로
        epoch (int): 에폭 번호 (랜덤 시드 변경용)
        general_sample_size (int): general 모드 샘플 수
        random_state (int): 랜덤 시드

    Returns:
        pd.DataFrame: 저장된 결과 DataFrame
    """
    # 데이터 불러오기
    df = pd.read_csv(input_csv_path)

    # 클러스터 딕셔너리 불러오기
    with open(cluster_path, 'r') as f:
        cluster_dict = json.load(f)

    # 클러스터에서 하나씩 샘플링한 pdb_name 수집
    random.seed(random_state + epoch)
    selected_pdbs = [random.choice(pdbs) for pdbs in cluster_dict.values() if pdbs]
    cluster_df = df[df['pdb_name'].isin(selected_pdbs) & (df['mode'] == 'ab')]

    # # general mode인 데이터 중 샘플링
    # general_df = df[df['mode'] == 'general']
    # sampled_general = general_df.sample(n=general_sample_size, random_state=random_state + epoch)

    # 병합
    # result_df = pd.concat([cluster_df, sampled_general])
    cluster_df.to_csv(output_csv_path, index=False)


def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 파일 또는 심볼릭 링크 제거
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 디렉토리 전체 제거
        except Exception as e:
            print(f"삭제 실패: {file_path} - {e}")

proj_name = "IPA_plddt_fm_hybrid_only_ab"
wandb.init(project='confidence_model_training', 
           name=f"{proj_name}",
           config={
    "learning_rate": 1e-4,
    "epochs": 1000,
    "batch_size": 1,
    "model_config": '/home/psh/protein-frame-flow/configs/model.yaml',
})
config = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir = os.path.join('/home/psh/protein-frame-flow/ckpt/confidence', proj_name)
os.makedirs(checkpoint_dir, exist_ok=True)

model_conf_path = '/home/psh/protein-frame-flow/configs/model.yaml'
model_conf = OmegaConf.load(model_conf_path)
model = ConfidenceModel(model_conf.model).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

class ConfidencePTDataset(Dataset):
    def __init__(self, pt_dir):
        self.pt_dir = pt_dir
        self.pt_files = sorted([
            os.path.join(pt_dir, f) for f in os.listdir(pt_dir)
            if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        
        data['input_for_confidence']['curr_trans'] = data['input_for_confidence']['curr_rigids'].get_trans()
        data['input_for_confidence']['curr_rotmats'] = data['input_for_confidence']['curr_rigids'].get_rots().get_rot_mats()
        del data['input_for_confidence']['curr_rigids']
        
        return data

val_dir = '/home/psh/protein-frame-flow/val_conf' # sampler로 미리 만들어놓기 
val_dataset = ConfidencePTDataset(val_dir)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

num_epochs = 150

for epoch in range(num_epochs):
    model.train()

    # sampling the trainset 

    csv_path = '/home/psh/data/train/merged_metadata_wt_nano.csv'
    sampled_csv_path = '/home/psh/protein-frame-flow/train_conf/metadata.csv'
    sample_csv(
        input_csv_path=csv_path,
        cluster_path='/home/psh/data/train/wantigen_70_loop_final_train.json',
        output_csv_path=sampled_csv_path,
        epoch=epoch,
        general_sample_size=10,
    )
    # initialize EvalRunner 
    inf_cfg = '/home/psh/protein-frame-flow/configs/_inference.yaml'
    inf_cfg = OmegaConf.load(inf_cfg)

    inf_cfg.inference.samples.csv_path = '/home/psh/protein-frame-flow/train_conf/metadata.csv'
    inf_cfg.inference.samples.samples_per_target = 1
    inf_cfg.inference.seed = epoch
    sampler = EvalRunner(inf_cfg)
    sampler.run_sampling(save_file=False)

    total_loss = 0
    total_batch = 0
    
    train_dir = '/home/psh/protein-frame-flow/train_conf' # sampler로 계속 업데이트트
    train_dataset = ConfidencePTDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    for i, batch in enumerate(train_loader):
        total_batch += 1
        # Dict 내부 텐서 GPU로
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        batch['input_for_confidence']['curr_rigids'] = du.create_rigid(batch['input_for_confidence']['curr_rotmats'],
                                                                       batch['input_for_confidence']['curr_trans'])
        input_for_confidence = batch['input_for_confidence']
        input_for_confidence = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in input_for_confidence.items()
        }

        node_mask = torch.ones_like(batch['diffuse_mask'], device=device)
        output = model(input_for_confidence, node_mask)

        # output shape = (L, num_bins) or (L,), match with target
        loss = lddt_loss(logits=output,
                                all_atom_pred_pos=batch["pred_positions"].squeeze(0), # predicted structure (b, l, 14, 3)
                                all_atom_positions=batch["atom14_gt_positions"], # gt stucture  (b, l, 14, 3)
                                all_atom_mask=batch["atom14_gt_exists"],
                                cdr_mask=batch['diffuse_mask']) # (b, l)

        print(f"Epoch {epoch+1}/{num_epochs} Step {i}, train loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss = total_loss / total_batch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "train_loss": total_loss})
    clear_directory(train_dir)

    # (선택) validation loop

    model.eval()
    val_loss = 0
    val_loss = 0
    total_val_batch = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            total_val_batch += 1
            # Dict 내부 텐서 GPU로
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            batch['input_for_confidence']['curr_rigids'] = du.create_rigid(batch['input_for_confidence']['curr_rotmats'],
                                                                        batch['input_for_confidence']['curr_trans'])
            input_for_confidence = batch['input_for_confidence']
            input_for_confidence = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in input_for_confidence.items()
            }

            node_mask = torch.ones_like(batch['diffuse_mask'], device=device)
            output = model(input_for_confidence, node_mask)

            # output shape = (L, num_bins) or (L,), match with target
            loss = lddt_loss(logits=output,
                                    all_atom_pred_pos=batch["pred_positions"].squeeze(0), # predicted structure (b, l, 14, 3)
                                    all_atom_positions=batch["atom14_gt_positions"], # gt stucture  (b, l, 14, 3)
                                    all_atom_mask=batch["atom14_gt_exists"],
                                    cdr_mask=batch['diffuse_mask']) # (b, l)
            val_loss += loss.item()
        val_loss = val_loss / total_val_batch
    print(f"  Validation Loss: {val_loss:.4f}")

    wandb.log({"epoch": epoch + 1, "val_loss": val_loss})

    # checkpoint 저장
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.ckpt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss,
        'val_loss': val_loss,
    }, checkpoint_path)