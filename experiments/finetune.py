import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data.datasets import ScopeDataset, PdbDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
from experiments import utils as eu
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class MaskingRatioCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        datamodule.current_epoch = trainer.current_epoch
        datamodule.masking_ratio = min(
            1.0, 
            datamodule.data_cfg.masking_scheduler.init_rate + 
            datamodule.data_cfg.masking_scheduler.masking_increase_ratio * trainer.current_epoch
        )
        log.info(f"Epoch {trainer.current_epoch}: masking_ratio = {datamodule.masking_ratio}")

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task
        self._setup_dataset()
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        self._train_device_ids = eu.get_available_device(self._exp_cfg.num_devices)
        log.info(f"Training with devices: {self._train_device_ids}")

        # FlowModule 초기화 (새로운 모듈 이미 포함)
        self._module: LightningModule = FlowModule(self._cfg)

        # Warm-start 체크포인트 로드
        if self._exp_cfg.warm_start:
            self._load_warm_start()

    def _load_warm_start(self):
        """체크포인트에서 가중치 로드 및 새로운 모듈 초기화"""
        checkpoint = torch.load(self._exp_cfg.warm_start, map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)  # 체크포인트 구조에 따라 조정
        flow_state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
        }

        # 기존 가중치 로드 (strict=False로 새로운 모듈 무시)
        missing_keys, unexpected_keys = self._module.model.load_state_dict(flow_state_dict)

        log.info(f"Loaded warm-start checkpoint from {self._exp_cfg.warm_start}")

        log.info(f"Missing keys (likely new modules): {missing_keys}")

        log.info(f"Unexpected keys: {unexpected_keys}")

    #     # 새로운 모듈의 가중치 초기화
    #     self._initialize_new_modules()

    # def _initialize_new_modules(self):
    #     """새로운 모듈의 가중치 초기화"""
    #     from torch import nn
    #     checkpoint = torch.load(self._exp_cfg.warm_start, map_location='cpu')
    #     state_dict = checkpoint.get("state_dict", checkpoint)
    #     checkpoint_keys = set(k.replace("model.", "") for k in state_dict.keys() if k.startswith("model."))

    #     for name, param in self._module.model.named_parameters():
    #         if name not in checkpoint_keys:
    #             log.info(f"Initializing new parameter: {name}")
    #             if param.dim() >= 2:  # Weight matrices
    #                 nn.init.xavier_uniform_(param)
    #             else:  # Biases
    #                 nn.init.zeros_(param)

    def _setup_dataset(self):
        if self._data_cfg.dataset == 'scope':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                ScopeDataset, self._cfg.scope_dataset, self._task)
        elif self._data_cfg.dataset == 'pdb':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task)
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}')

    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
            callbacks.append(MaskingRatioCallback())
        else:
            logger = WandbLogger(**self._exp_cfg.wandb)
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            callbacks.append(MaskingRatioCallback())

            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
            gradient_clip_val=0.5
        )

        # 체크포인트 기반으로 훈련 재개
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            # ckpt_path=self._exp_cfg.warm_start if self._exp_cfg.warm_start else None
        )

@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Merge warm start config with current config
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(warm_start_cfg.model, cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded and merged warm start config from {warm_start_cfg_path}')

    # Experiment 초기화 및 훈련
    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()