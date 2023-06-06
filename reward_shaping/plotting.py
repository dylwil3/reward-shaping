# type:ignore
from typing import DefaultDict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def make_distribution_gif(df: pd.DataFrame):
    ...


def plot_evals(df: pd.DataFrame):
    """Gives Axes object for plotting average training run with shaded std.

    Arg
        df: Pandas DataFrame with columns corresponding to runs and row i
        corresponding to ith evaluation during the run.
    Returns
        Matplotlib Axes object which will display average training
        rewturns and a shaded region going one standard deviation
        above and below.

    Warning: At the moment the DataManager saves dataframe as *transpose*
    of what we want to input in this function.
    """
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    fig, ax = plt.subplots()
    ax.plot(df["mean"])  # pyright:ignore
    ax.fill_between(  # pyright:ignore
        np.arange(len(df)), df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.5
    )
    return fig, ax


def default_dict_to_list(d: DefaultDict[int, float], n: int):
    """Converts defaultdictionary into list of values, including omitted/default values."""
    e = [d[i] for i in range(n**2)]
    return e


def update_dict_to_array(v: List[DefaultDict[int, float]], n: int) -> np.ndarray:
    """Converts list of defaultdictionaries to square array of indicated size."""
    runs = len(v)
    if runs == 0:
        raise ValueError("List of dictionaries must be nonempty.")
    elif n <= 0:
        raise ValueError("Must have positive map size.")
    else:
        e = [default_dict_to_list(v[run], n) for run in range(runs)]
        flattened = np.array(e)
        return flattened


def filter_inf_start_value(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separates out runs that updated start value in finite time.

    Returns tuple tup of arrays. tup[0] has rows corresponding to
    successful runs (finite update time) and tup[0] has rows corresponding
    to unsuccessful runs (never updated start value.)
    """
    msk_inf = flat[:, 0] == float("inf")
    msk_fin = flat[:, 0] != float("inf")
    finite_start = flat[msk_fin]
    infinite_start = flat[msk_inf]
    return finite_start, infinite_start


def visualize_frozen_updates(
    v: List[DefaultDict[int, float]], n: int
) -> tuple[plt.Axes, int, int]:
    """Heatmap visualizing the timestep at which values of states are updated during training.

    Arg
        List of (default)dictionaries. Each element of the list corresponds to a run, each
        dictionary records the timestep at which a state's value was updated in training,
        with float("inf") meaning the value was never updated.
    Returns
        Tuple consisting of...
            0. Matplotlib Axes object giving heatmap of average timestep for update amongst
            all runs where the start value was updated.
            1. Number of runs with finite update timestep for start value.
            2. Total number of runs (i.e. length of original list).
    """
    flattened = update_dict_to_array(v, n)
    finite, _ = filter_inf_start_value(flattened)
    averaged = np.mean(finite, axis=0)
    reshaped = averaged.reshape((n, n))
    ax = sns.heatmap(
        reshaped,
        annot=True,
        fmt=".0f",
        linewidths=0.5,  # pyright:ignore
        cmap="winter_r",
    )
    ax.set_title("Timesteps before Updating Value")
    ax.axis("off")
    return ax, len(finite), len(flattened)
