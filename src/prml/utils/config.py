"""YAML configuration loading utilities."""

from pathlib import Path

import yaml


def load_config(path):
    """Load a YAML configuration file.

    Parameters
    ----------
    path : path_like
        Path to the YAML document.

    Returns
    -------
    object
        Python object produced by :func:`yaml.safe_load`.
    """
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)
