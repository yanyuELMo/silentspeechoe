"""Model factory — builds models from Hydra configuration.

Keeps the wiring simple: a single public function inspects the config
and instantiates the requested model.
"""

from __future__ import annotations

from omegaconf import DictConfig

from .bone_cnn import BoneBinauralCNN


def build_model(cfg: DictConfig) -> BoneBinauralCNN:
    """Build a model according to the Hydra config.

    The ``model`` config group is expected to provide at least ``name``.
    Additional keys (e.g. ``conv1_channels``) are forwarded as kwargs.

    Currently supported model names:

    * ``bone_binaural`` — :class:`BoneBinauralCNN`
    """
    model_cfg = cfg.model
    name = model_cfg.name

    if name == "bone_binaural":
        kwargs: dict = {}
        # Forward optional model-config fields if present
        for key in (
            "in_channels",
            "num_classes",
            "conv1_channels",
            "conv2_channels",
            "kernel_size_1",
            "kernel_size_2",
            "dropout",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        return BoneBinauralCNN(**kwargs)

    raise ValueError(f"Unknown model name: {name}")
