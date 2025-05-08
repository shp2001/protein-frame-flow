import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.datasets import ScopeDataset, PdbDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
from experiments import utils as eu
import wandb


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class MaskingRatioCallback(Callback):
    # @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        datamodule._train_dataset.set_current_epoch(trainer.current_epoch)
        datamodule._train_dataset.current_epoch = trainer.current_epoch
        datamodule.set_current_epoch(trainer.current_epoch)
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
        self._module: LightningModule = FlowModule(self._cfg)

        if self._exp_cfg.add_modules:
            # 기존 모델 weight 로드
            state_dict = torch.load(cfg.experiment.warm_start, map_location='cpu')["state_dict"]
            # 필요한 부분만 추출해서 로드
            flow_state_dict = {k.replace("flow.", ""): v for k, v in state_dict.items() if k.startswith("flow.")}

            # FlowModule에만 로드
            self._module.model.load_state_dict(flow_state_dict, strict=False)
            for name, param in self._module.model.named_parameters():

                log.info(f"Found prmsd param: {name}")

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
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            callbacks.append(MaskingRatioCallback())
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
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
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )
        # trainer.fit(
        #     model=self._module,
        #     datamodule=self._datamodule,
        # )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
