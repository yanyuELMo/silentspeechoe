"""Model factory — builds models from Hydra configuration.

Keeps the wiring simple: a single public function inspects the config
and instantiates the requested model.
"""

from __future__ import annotations

from omegaconf import DictConfig

from .bone_cnn import BoneBinauralCNN
from .bone_tcn import BoneRawTCN
from .imu_cnn import IMUCNN, IMUDoubleCNN
from .imu_mlp import IMUMLP, MFCCMLP, IMUMLPArcFace


def build_model(
    cfg: DictConfig,
) -> (
    BoneBinauralCNN
    | BoneRawTCN
    | IMUCNN
    | IMUDoubleCNN
    | IMUMLP
    | IMUMLPArcFace
    | MFCCMLP
):
    """Build a model according to the Hydra config.

    The ``model`` config group is expected to provide at least ``name``.
    Additional keys (e.g. ``conv1_channels``) are forwarded as kwargs.

    Currently supported model names:

    * ``bone_binaural`` — :class:`BoneBinauralCNN`
    * ``bone_raw_tcn`` — :class:`BoneRawTCN`
    * ``imu_cnn`` — :class:`IMUCNN`
    * ``imu_double_cnn`` — :class:`IMUDoubleCNN`
    * ``mfcc_mlp`` — :class:`MFCCMLP`
    * ``imu_mlp`` — :class:`IMUMLP`
    * ``imu_mlp_arcface`` — :class:`IMUMLPArcFace`
    """
    model_cfg = cfg.model
    name = model_cfg.name

    if name == "bone_binaural":
        kwargs: dict = {}
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

    if name == "bone_raw_tcn":
        kwargs: dict = {}
        for key in (
            "in_channels",
            "hidden_channels",
            "num_classes",
            "kernel_size",
            "dilations",
            "dropout",
        ):
            if key in model_cfg:
                val = model_cfg[key]
                # Hydra stores lists as ListConfig — convert to tuple.
                if key == "dilations" and not isinstance(val, tuple):
                    val = tuple(int(d) for d in val)
                kwargs[key] = val
        return BoneRawTCN(**kwargs)

    if name == "imu_cnn":
        kwargs: dict = {}
        for key in (
            "in_channels",
            "num_classes",
            "conv1_channels",
            "conv2_channels",
            "conv3_channels",
            "kernel_size_1",
            "kernel_size_2",
            "kernel_size_3",
            "dropout",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        return IMUCNN(**kwargs)

    if name == "mfcc_mlp":
        kwargs: dict = {}
        for key in (
            "in_features",
            "num_classes",
            "hidden1",
            "hidden2",
            "dropout",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        return MFCCMLP(**kwargs)

    if name == "imu_mlp":
        kwargs: dict = {}
        for key in (
            "in_features",
            "num_classes",
            "hidden1",
            "hidden2",
            "dropout",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        return IMUMLP(**kwargs)

    if name == "imu_mlp_arcface":
        kwargs: dict = {}
        for key in (
            "in_features",
            "num_classes",
            "hidden1",
            "hidden2",
            "dropout",
            "arcface_scale",
            "arcface_margin",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        return IMUMLPArcFace(**kwargs)

    if name in {"imu_double_cnn", "imu_feature_cnn"}:
        kwargs: dict = {}
        for key in (
            "in_features",
            "num_classes",
            "conv1_channels",
            "conv2_channels",
            "conv3_channels",
            "kernel_size_1",
            "kernel_size_2",
            "kernel_size_3",
            "dropout",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        return IMUDoubleCNN(**kwargs)

    raise ValueError(f"Unknown model name: {name}")
