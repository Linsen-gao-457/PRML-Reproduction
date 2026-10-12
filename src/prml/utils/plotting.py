"""Helpers for saving Matplotlib figures."""

from pathlib import Path


def save_figure(figure, path, *, dpi=150):
    """Save a figure and create its parent directory if needed.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    path : path_like
        Destination image path.
    dpi : int or float, default=150
        Resolution in dots per inch.

    Returns
    -------
    pathlib.Path
        Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
