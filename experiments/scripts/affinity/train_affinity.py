import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data.datasets_affinity import PdbDataset
from data.protein_dataloader_affinity import ProteinData
from models.affinity_module import AffinityModule
from experiments import utils as eu
import wandb


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


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

        self._module: LightningModule = AffinityModule(self._cfg)

    def _setup_dataset(self):
        self._train_dataset, self._valid_dataset = eu.dataset_creation(
            PdbDataset, self._cfg.pdb_dataset, self._task)

    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._data_cfg.loader.num_workers = 0
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
            callbacks.append(LearningRateMonitor(logging_interval='step'))
            # Save config only for main process.

            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        # Check if a warm start checkpoint is provided to load weights
        if self._exp_cfg.warm_start and os.path.exists(self._exp_cfg.warm_start):
            log.info(f"Loading weights from checkpoint: {self._exp_cfg.warm_start}")
            self._module = AffinityModule.load_from_checkpoint(
                checkpoint_path=self._exp_cfg.warm_start,
                cfg=self._cfg,
                strict=False
            )
        else:
            log.info("No warm start checkpoint found. Training from scratch.")

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            strategy=DDPStrategy(find_unused_parameters=True),
            devices=self._exp_cfg.num_devices,
            gradient_clip_val=1.0
        )

        # trainer.fit(
        #     model=self._module,
        #     datamodule=self._datamodule,
        #     ckpt_path=self._exp_cfg.warm_start,
        # )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
        )


@hydra.main(version_base=None, config_path="../../../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
