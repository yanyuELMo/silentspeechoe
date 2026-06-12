"""Configuration file tests for the scaffold."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


def test_yaml_files_are_parseable() -> None:
    """Ensure all placeholder YAML files remain valid."""

    yaml_files = sorted(CONFIG_ROOT.rglob("*.yaml"))
    assert yaml_files, "No YAML configuration files were found."

    for yaml_file in yaml_files:
        with yaml_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert isinstance(data, dict), f"{yaml_file} did not parse into a mapping."


def test_root_hydra_config_has_defaults() -> None:
    """Ensure the root config follows Hydra defaults composition style."""

    config_path = CONFIG_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    defaults = config.get("defaults")
    assert isinstance(defaults, list)
    assert "_self_" in defaults
