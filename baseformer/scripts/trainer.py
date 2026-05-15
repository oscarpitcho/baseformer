"""
Trainer class for language model training with checkpointing and wandb logging.
"""

import logging
from pathlib import Path
from typing import Callable

import numpy.typing as npt
import torch
from omegaconf import DictConfig, OmegaConf

from baseformer.data.dataset import get_batch
from baseformer.nn.utils import loss_cross_ent, perplexity, clip_gradients_
from baseformer.tokenization.bpe import BPETokenizer

try:
    import wandb
except ImportError:
    wandb = None

from baseformer.optim.adamw import AdamW

log = logging.getLogger(__name__)

# Sample prompts for text generation during evaluation
SAMPLE_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "One day, a",
]


class Trainer:
    """
    Trainer for language model training.

    Handles the training loop, evaluation, checkpointing, and wandb logging.
    Designed to be extended for DDP support.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: AdamW,
        lr_schedule: Callable[[int, float], float],
        train_data: npt.NDArray,
        val_data: npt.NDArray | None,
        cfg: DictConfig,
    ):
        """
        Args:
            model: The language model to train.
            optimizer: AdamW optimizer instance.
            lr_schedule: Learning rate schedule callable (step, base_lr) -> lr.
            train_data: Training data as memory-mapped numpy array.
            val_data: Validation data as memory-mapped numpy array (optional).
            cfg: Full Hydra config.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.step = 0
        self.tokens_processed = 0
        self.device = cfg.device
        self.batch_size = cfg.experiment.batch_size
        self.context_length = cfg.data.sequence_length

        # Load tokenizer for sample generation
        self.tokenizer = BPETokenizer.from_files(
            cfg.data.tokenizer.vocab_path,
            cfg.data.tokenizer.merges_path,
        )

    def train(self) -> None:
        """Main training loop."""
        self.step = self._load_checkpoint()

        self.model.train()

        log.info(f"Starting training from step {self.step}")

        while self.step < self.cfg.experiment.max_steps:
            x, y = get_batch(self.train_data, self.batch_size, self.context_length, self.device)
            loss, num_tokens = self._train_step(x, y)
            self.tokens_processed += num_tokens

            # Logging
            if self.step % 10 == 0:
                curr_lr = self.lr_schedule(self.step, self.optimizer.lr)
                self._wandb_log({"train/loss": loss, "train/lr": curr_lr, "train/tokens": self.tokens_processed})
                log.info(f"Step {self.step}: loss={loss:.4f}, lr={curr_lr:.6e}, tokens={self.tokens_processed:,}")


            # Evaluation
            if (self.val_data is not None) and (self.step % self.cfg.experiment.eval_every == 0) and (self.step > 0):
                log.info(f"Running evaluation at step {self.step}")

                eval_metrics = self._eval()
                self._wandb_log({f"eval/{k}": v for k, v in eval_metrics.items()})
                log.info(f"Step {self.step}: eval_loss={eval_metrics['loss']:.4f}")

                # Generate and log text samples
                samples = self._generate_samples()
                self._log_samples(samples)

                self.model.train()

            # Checkpointing
            if self.step % self.cfg.checkpoint.save_every == 0 and self.step > 0:
                self._save_checkpoint()

            self.step += 1

        # Final checkpoint
        self._save_checkpoint()
        log.info("Training complete.")

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, int]:
        """Execute a single training step.

        Args:
            x: Input token ids, already on device.
            y: Target labels, already on device.

        Returns:
            Tuple of (loss value, number of tokens in batch).
        """
        # Forward pass
        logits = self.model(x)

        # Compute cross-entropy loss
        loss = loss_cross_ent(logits, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.cfg.experiment.grad_clip > 0:
            clip_gradients_(self.model.parameters(), self.cfg.experiment.grad_clip)

        # Optimizer step with LR schedule
        self.optimizer.step(self.lr_schedule)

        return loss.item(), y.numel()

    @torch.no_grad()
    def _eval(self) -> dict:
        """Run evaluation on validation set.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_ppl = 0.0
        num_batches = 10

        for _ in range(num_batches):
            x, y = get_batch(self.val_data, self.batch_size, self.context_length, self.device)
            logits = self.model(x)
            total_loss += loss_cross_ent(logits, y).item()
            total_ppl += perplexity(logits, y).item()

        return {"loss": total_loss / num_batches, "perplexity": total_ppl / num_batches}

    @torch.no_grad()
    def _generate_samples(self, max_tokens: int = 100) -> list[dict]:
        """Generate text samples from the model.

        Args:
            max_tokens: Maximum tokens to generate per sample.

        Returns:
            List of dicts with 'prompt' and 'generated' keys.
        """
        self.model.eval()
        samples = []
        eos_token = self.tokenizer.encode("<|endoftext|>")[0]
        temperature = self.cfg.experiment.temperature
        top_p = self.cfg.experiment.top_p

        for prompt in SAMPLE_PROMPTS:
            # Encode prompt
            prompt_ids = self.tokenizer.encode(prompt)
            token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            # Generate tokens using model.decode iterator, stop at EOS or max tokens
            generated_ids = list(prompt_ids)
            for i, next_token in enumerate(self.model.decode(token_ids, temperature=temperature, top_p=top_p)):
                generated_ids.append(next_token[0].item())
                if next_token == eos_token or i >= max_tokens:
                    break

            # Decode full sequence
            generated_text = self.tokenizer.decode(generated_ids)
            samples.append({"prompt": prompt, "generated": generated_text})

        return samples

    def _log_samples(self, samples: list[dict]) -> None:
        """Log generated samples to console and wandb."""
        log.info("=" * 60)
        log.info("Generated samples:")
        log.info("=" * 60)

        wandb_text = []
        for i, sample in enumerate(samples):
            log.info(f"[{i+1}] Prompt: {sample['prompt']!r}")
            log.info(f"    Generated: {sample['generated']!r}")
            log.info("-" * 40)
            wandb_text.append(f"**Prompt:** {sample['prompt']}\n\n**Generated:** {sample['generated']}")

        # Log to wandb as a table
        if self.cfg.wandb.enabled and wandb is not None:
            table = wandb.Table(columns=["prompt", "generated"])
            for sample in samples:
                table.add_data(sample["prompt"], sample["generated"])
            wandb.log({"eval/samples": table}, step=self.step)

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        ckpt_dir = Path(self.cfg.checkpoint.dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / f"step_{self.step}.pt"

        torch.save(
            {
                "step": self.step,
                "tokens_processed": self.tokens_processed,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            ckpt_path,
        )
        log.info(f"Saved checkpoint to {ckpt_path}")

    def _load_checkpoint(self) -> int:
        """Load checkpoint if available.

        Returns:
            Starting step number.
        """
        start_step = 0

        # Initialize weights from another experiment (transfer learning)
        if self.cfg.checkpoint.init_from:
            ckpt = torch.load(self.cfg.checkpoint.init_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            log.info(f"Initialized weights from {self.cfg.checkpoint.init_from}")

        # Resume full training state
        resume_path = self._resolve_resume_path()
        if resume_path:
            ckpt = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            self.tokens_processed = ckpt.get("tokens_processed", 0)
            log.info(f"Resumed from {resume_path} at step {start_step}, tokens={self.tokens_processed:,}")

        return start_step

    def _resolve_resume_path(self) -> str | None:
        """Find checkpoint to resume from.

        Returns:
            Path to checkpoint file, or None if no checkpoint found.
        """
        resume = self.cfg.checkpoint.resume

        if resume is None:
            return None

        if resume == "auto":
            ckpt_dir = Path(self.cfg.checkpoint.dir)
            if not ckpt_dir.exists():
                return None

            # Find all step_*.pt files
            ckpts = list(ckpt_dir.glob("step_*.pt"))
            if not ckpts:
                return None

            # Sort by step number and return latest
            ckpts.sort(key=lambda p: int(p.stem.split("_")[1]))
            return str(ckpts[-1])

        # Explicit path provided
        return resume

    def _wandb_log(self, metrics: dict) -> None:
        """Log metrics to wandb if enabled."""
        if self.cfg.wandb.enabled and wandb is not None:
            wandb.log(metrics, step=self.step)

