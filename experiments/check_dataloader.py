import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm # tqdm 추가

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data.datasets import PdbDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
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
        
        # dry_run 모드가 아닐 때만 모델을 로드하여 시간 절약
        if not self._cfg.get("dry_run", False):
            self._module: LightningModule = FlowModule(self._cfg)

    def _setup_dataset(self):
        self._train_dataset, self._valid_dataset = eu.dataset_creation(
            PdbDataset, self._cfg.pdb_dataset, self._task)

    def dry_run(self):
        """
        모델 학습 없이 DataLoader만 순회하며 데이터 무결성 검사
        """
        log.info("========================================")
        log.info("       STARTING DATALOADER DRY RUN      ")
        log.info("========================================")

        # 1. 디버깅을 위해 워커를 0으로 설정 (에러 추적 용이)
        if self._data_cfg.loader.num_workers > 0:
            log.info(f"Forcing num_workers to 0 for debugging (Original: {self._data_cfg.loader.num_workers})")
            self._data_cfg.loader.num_workers = 0

        # 2. DataModule 준비
        self._datamodule.prepare_data()
        self._datamodule.setup(stage='fit')
        
        train_loader = self._datamodule.train_dataloader()
        
        log.info(f"Total Batches to check: {len(train_loader)}")

        # 3. 데이터 순회 (모델 연산 X)
        try:
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Checking Data"):
                # 필요한 경우 여기서 텐서 shape 등을 간단히 assert 할 수 있습니다.
                # 예: assert batch['pos'].shape[0] > 0
                pass
                
        except Exception as e:
            log.error(f"\n[CRITICAL ERROR] Failed at Batch Index: {i}")
            log.error("This usually indicates a corrupted file or preprocessing error in this batch.")
            log.error(f"Error Message: {e}")
            
            # 여기서 batch 정보를 출력하면 어떤 파일인지 힌트를 얻을 수 있음 (Dataset 구현에 따라 다름)
            raise e

        log.info("✅ Dry Run Completed! No dataloader errors found.")

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
            self._module = FlowModule.load_from_checkpoint(
                checkpoint_path=self._exp_cfg.warm_start,
                cfg=self._cfg,
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
            strategy='ddp',
            devices=self._exp_cfg.num_devices,
            gradient_clip_val=1.0
        )

        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):
    exp = Experiment(cfg=cfg)
    
    # 커맨드라인에서 +dry_run=True 를 주거나 config에 dry_run: True가 있으면 실행
    if cfg.get("dry_run", False):
        exp.dry_run()
    else:
        exp.train()

if __name__ == "__main__":
    main()