import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from models.flow_model import ConfidenceModel
from models.loss import compute_prmsd_loss, lddt_loss
from omegaconf import OmegaConf

import data.utils as du 
import os 
import wandb  # wandb 추가

from experiments.inference_se3_flows import EvalRunner

wandb.init(project='confidence_model_training', 
           name="IPA_plddt",
           config={
    "learning_rate": 1e-4,
    "epochs": 1000,
    "batch_size": 1,
    "model_config": '/home/psh/protein-frame-flow/configs/model.yaml',
})
config = wandb.config
inf_cfg = '/home/psh/protein-frame-flow/configs/_inference.yaml'
sampler = EvalRunner(OmegaConf(inf_cfg))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

val_path = pass
val_dataset = ConfidencePTDataset(val_path)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_batch = 0
    sampler.run_sampling()
    train_path = pass
    train_dataset = ConfidencePTDataset(train_path)
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
    # (선택) validation loop

    # model.eval()
    # val_loss = 0
    # val_loss = 0
    # total_batch = 0
    # with torch.no_grad():
    #     for i, batch in enumerate(val_loader):
    #         total_batch += 
    #         # Dict 내부 텐서 GPU로
    #         batch = {
    #             k: v.to(device) if isinstance(v, torch.Tensor) else v
    #             for k, v in batch.items()
    #         }

    #         batch['input_for_confidence']['curr_rigids'] = du.create_rigid(batch['input_for_confidence']['curr_rotmats'],
    #                                                                     batch['input_for_confidence']['curr_trans'])
    #         input_for_confidence = batch['input_for_confidence']
    #         input_for_confidence = {
    #             k: v.to(device) if isinstance(v, torch.Tensor) else v
    #             for k, v in input_for_confidence.items()
    #         }

    #         node_mask = torch.ones_like(batch['diffuse_mask'], device=device)
    #         output = model(input_for_confidence, node_mask)

    #         # output shape = (L, num_bins) or (L,), match with target
    #         loss = compute_prmsd_loss(logits=output,
    #                                 all_atom_pred_pos=batch["pred_positions"].squeeze(0), # predicted structure (b, l, 14, 3)
    #                                 all_atom_positions=batch["atom14_gt_positions"], # gt stucture  (b, l, 14, 3)
    #                                 all_atom_mask=batch["atom14_gt_exists"],
    #                                 cdr_mask=batch['diffuse_mask']) # (b, l)
    #         val_loss += loss.item()
    # print(f"  Validation Loss: {val_loss:.4f}")

    # wandb.log({"epoch": epoch + 1, "val_loss": val_loss})