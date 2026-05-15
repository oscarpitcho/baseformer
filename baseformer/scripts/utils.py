from pathlib import Path
from omegaconf import OmegaConf
from baseformer.nn.transformer import TransformerLM
from baseformer.nn.position import RotaryPositionalEmbedding


# This is just a placeholder to instantiate the model
# This does not related to a tokenizer
VOCAB_SIZE = 10048


def load_model(model_name: str, device: str = "cuda") -> TransformerLM:
    """Load model from config files."""
    config_dir = Path(__file__).parent.parent / "configs"

    base_cfg = OmegaConf.load(config_dir / "config.yaml")
    model_cfg = OmegaConf.load(config_dir / "model" / f"{model_name}.yaml")
    position_cfg = OmegaConf.load(config_dir / "position" / "rope.yaml")

    cfg = OmegaConf.merge(
        base_cfg,
        {"model": model_cfg},
        {"position": position_cfg},
        {"model": {"vocab_size": VOCAB_SIZE}},
    )

    d_k = cfg.model.d_model // cfg.model.num_heads
    rope = RotaryPositionalEmbedding(
        theta=cfg.position.theta,
        d_k=d_k,
        max_seq_len=cfg.position.max_seq_len,
        device=device,
    )

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope=rope,
        device=device,
    )
    return model.to(device)

