"""
Main entry point for training with Hydra configuration.

Usage:
    python -m baseformer.main experiment_name=my_run data.train_path=data/owt_train_tokens.npz data.val_path=data/owt_valid_tokens.npz

    # Override experiment params
    python -m baseformer.main experiment.lr=3e-4 experiment.batch_size=64

    # Resume from checkpoint
    python -m baseformer.main checkpoint.resume=outputs/my_exp/checkpoints/step_5000.pt

    # Transfer weights from another experiment
    python -m baseformer.main experiment_name=finetune checkpoint.init_from=outputs/pretrain/checkpoints/step_10000.pt

    # Debug mode
    uv run python -m debugpy --listen 1234 --wait-for-client main.py --config-name wandb=disabled

Reusable functions:
    initialize_data(cfg) -> (train_loader, val_loader)
    initialize_training(cfg) -> (model, optimizer, lr_schedule, rope)
"""

import logging
from functools import partial
from typing import Callable

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from baseformer.nn.transformer import TransformerLM
from baseformer.nn.position import RotaryPositionalEmbedding
from baseformer.optim.adamw import AdamW
from baseformer.optim.scheduler import get_lr_cosine_schedule
from baseformer.data.dataset import TokenizedDataset
from baseformer.trainer import Trainer

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


def initialize_data(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Load and prepare train/val dataloaders.

    Args:
        cfg: Hydra config with data.train_path, data.val_path, 
             data.sequence_length, experiment.batch_size.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = TokenizedDataset(cfg.data.train_path, cfg.data.sequence_length)
    val_dataset = TokenizedDataset(cfg.data.val_path, cfg.data.sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.experiment.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, val_loader


def initialize_training(
    cfg: DictConfig,
) -> tuple[TransformerLM, AdamW, Callable[[int, float], float], RotaryPositionalEmbedding]:
    """Initialize model, optimizer, lr_schedule, and RoPE.

    Sets random seed, builds RoPE, model, optimizer with weight decay groups,
    and LR schedule. All training components in one place.

    Args:
        cfg: Hydra config with seed, model.*, position.*, experiment.*, device.

    Returns:
        Tuple of (model, optimizer, lr_schedule, rope).
    """
    # Set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Build RoPE
    d_k = cfg.model.d_model // cfg.model.num_heads
    rope = RotaryPositionalEmbedding(
        theta=cfg.position.theta,
        d_k=d_k,
        max_seq_len=cfg.position.max_seq_len,
        device=cfg.device,
    )

    # Build model
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope=rope,
        device=cfg.device,
    )
    model = model.to(cfg.device)

    # Build optimizer with weight decay only on non-bias, non-norm params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.experiment.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        param_groups,
        lr=cfg.experiment.lr,
        betas=(cfg.experiment.beta1, cfg.experiment.beta2),
    )

    # Build LR schedule
    lr_schedule = partial(
        get_lr_cosine_schedule,
        min_learning_rate=cfg.experiment.min_lr,
        warmup_iters=cfg.experiment.warmup_steps,
        cosine_cycle_iters=cfg.experiment.max_steps,
    )

    return model, optimizer, lr_schedule, rope


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point: logging, wandb, orchestration."""
    # Log resolved config
    log.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Initialize wandb
    if cfg.wandb.enabled:
        if wandb is None:
            log.warning("wandb not installed, disabling logging")
        else:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            override_dirname = hydra_cfg.job.override_dirname
            if override_dirname:
                run_name = f"{cfg.experiment_name}/{override_dirname}"
            else:
                run_name = cfg.experiment_name

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name or run_name,
                tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
                config=OmegaConf.to_container(cfg, resolve=True)    
            )

    # Initialize components
    train_loader, val_loader = initialize_data(cfg)
    model, optimizer, lr_schedule, rope = initialize_training(cfg)

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params:,}")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
    )
    trainer.train()

    # Cleanup
    if cfg.wandb.enabled and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
