"""Model factory for baseline model entry points."""

from __future__ import annotations

from omegaconf import DictConfig

from .cnn import IMUBinauralCNN, IMUBinauralFeatureCNN, SensorCNN
from .feature_token_transformer import (
    FeatureTokenTransformer,
    IMUBinauralFeatureTokenTransformer,
)
from .lstm import IMUBinauralBiLSTM, SensorBiLSTM
from .mlp import FeatureMLP, IMUBinauralFeatureMLP, IMUFeatureMLP
from .resnet import IMUBinauralResNet, SensorResNet
from .tcn import (
    BoneAccTemporalCNN,
    BoneRawTCN,
    IMUBinauralLRDiffTemporalCNN,
    IMUTemporalCNN,
)


def build_model(
    cfg: DictConfig,
) -> (
    BoneAccTemporalCNN
    | BoneRawTCN
    | IMUBinauralLRDiffTemporalCNN
    | IMUTemporalCNN
    | FeatureMLP
    | FeatureTokenTransformer
    | SensorBiLSTM
    | SensorCNN
    | SensorResNet
):
    """Build a model according to the Hydra config."""
    model_cfg = cfg.model
    name = model_cfg.name

    if name == "bone_acc_temporal_cnn":
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
                value = model_cfg[key]
                if key == "dilations" and not isinstance(value, tuple):
                    value = tuple(int(d) for d in value)
                kwargs[key] = value
        return BoneAccTemporalCNN(**kwargs)

    if name in (
        "imu_temporal_cnn",
        "imu_temporal_cnn_binaural",
        "imu_binaural_lrdiff_temporal_cnn",
    ):
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
                value = model_cfg[key]
                if key == "dilations" and not isinstance(value, tuple):
                    value = tuple(int(d) for d in value)
                kwargs[key] = value
        if name == "imu_binaural_lrdiff_temporal_cnn":
            return IMUBinauralLRDiffTemporalCNN(**kwargs)
        return IMUTemporalCNN(**kwargs)

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
                value = model_cfg[key]
                if key == "dilations" and not isinstance(value, tuple):
                    value = tuple(int(d) for d in value)
                kwargs[key] = value
        return BoneRawTCN(**kwargs)

    if name in ("sensor_bilstm", "imu_binaural_bilstm"):
        kwargs: dict = {}
        for key in (
            "in_channels",
            "hidden_size",
            "num_layers",
            "num_classes",
            "dropout",
            "bidirectional",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        if name == "imu_binaural_bilstm":
            return IMUBinauralBiLSTM(**kwargs)
        return SensorBiLSTM(**kwargs)

    if name in ("feature_mlp", "imu_feature_mlp", "imu_feature_mlp_binaural"):
        kwargs: dict = {}
        for key in (
            "in_features",
            "hidden_features",
            "num_classes",
            "dropout",
            "use_batch_norm",
        ):
            if key in model_cfg:
                value = model_cfg[key]
                if key == "hidden_features" and not isinstance(value, tuple):
                    value = tuple(int(width) for width in value)
                kwargs[key] = value
        if name == "imu_feature_mlp":
            return IMUFeatureMLP(**kwargs)
        if name == "imu_feature_mlp_binaural":
            return IMUBinauralFeatureMLP(**kwargs)
        return FeatureMLP(**kwargs)

    if name in ("sensor_cnn", "imu_binaural_cnn", "imu_binaural_feature_cnn"):
        kwargs: dict = {}
        for key in (
            "in_channels",
            "hidden_channels",
            "num_classes",
            "kernel_size",
            "num_conv_blocks",
            "dropout",
            "return_features_for_training",
        ):
            if key in model_cfg:
                value = model_cfg[key]
                if key == "num_conv_blocks" and not isinstance(value, int):
                    value = int(value)
                kwargs[key] = value
        if name == "imu_binaural_cnn":
            return IMUBinauralCNN(**kwargs)
        if name == "imu_binaural_feature_cnn":
            return IMUBinauralFeatureCNN(**kwargs)
        return SensorCNN(**kwargs)

    if name in ("sensor_resnet", "imu_binaural_resnet"):
        kwargs: dict = {}
        for key in (
            "in_channels",
            "base_channels",
            "num_classes",
            "kernel_size",
            "num_stages",
            "blocks_per_stage",
            "dropout",
        ):
            if key in model_cfg:
                value = model_cfg[key]
                if key in ("num_stages", "blocks_per_stage"):
                    kwargs[key] = int(value)
                else:
                    kwargs[key] = value
        if name == "imu_binaural_resnet":
            return IMUBinauralResNet(**kwargs)
        return SensorResNet(**kwargs)

    if name in (
        "feature_token_transformer",
        "imu_feature_token_transformer_binaural",
    ):
        kwargs: dict = {}
        for key in (
            "in_features",
            "num_tokens",
            "token_dim",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "mlp_ratio",
            "embedding_dim",
            "num_classes",
            "dropout",
        ):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]
        if name == "imu_feature_token_transformer_binaural":
            return IMUBinauralFeatureTokenTransformer(**kwargs)
        return FeatureTokenTransformer(**kwargs)

    raise ValueError(
        f"Unknown model name: {name!r}. "
        "Expected one of 'bone_acc_temporal_cnn', 'imu_temporal_cnn', "
        "'imu_temporal_cnn_binaural', 'imu_binaural_lrdiff_temporal_cnn', "
        "'bone_raw_tcn', 'sensor_bilstm', 'imu_binaural_bilstm', 'sensor_cnn', "
        "'imu_binaural_cnn', 'imu_binaural_feature_cnn', "
        "'sensor_resnet', 'imu_binaural_resnet', "
        "'feature_mlp', 'imu_feature_mlp', 'imu_feature_mlp_binaural', "
        "'feature_token_transformer', or "
        "'imu_feature_token_transformer_binaural'."
    )
