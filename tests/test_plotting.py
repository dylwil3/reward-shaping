from collections import defaultdict
from typing import DefaultDict

import numpy as np
import pandas as pd

from reward_shaping.plotting import (
    default_dict_to_list,
    filter_inf_start_value,
    plot_evals,
    update_dict_to_array,
    visualize_frozen_updates,
)


def test_default_dict_to_list():
    dd: DefaultDict[int, float] = defaultdict(lambda: float("inf"))
    dd[0] = 1
    dd[1] = 2
    dd[2] = 3
    e = default_dict_to_list(dd, 2)
    assert e[0] == 1
    assert e[1] == 2
    assert e[2] == 3
    assert e[3] == float("inf")
    assert len(e) == 4


def test_update_dict_to_array():
    dd = defaultdict(lambda: float("inf"))
    dd[0] = 1
    dd[1] = 2
    dd[2] = 3
    ff = defaultdict(lambda: float("inf"))
    ff[0] = 10
    ff[1] = 9
    v = [dd, ff]
    flat = update_dict_to_array(v, 2)
    assert len(flat[0]) == 4
    assert len(flat[1]) == 4
    assert len(flat) == 2
    assert flat[0][3] == float("inf")
    assert flat[1][2] == float("inf")
    assert flat[1][0] == 10
    assert flat[0][1] == 2
    assert isinstance(flat, np.ndarray)
    assert isinstance(flat[0], np.ndarray)


def test_visualize_frozen_updates():
    dd = defaultdict(lambda: float("inf"))
    dd[0] = 1
    dd[1] = 2
    dd[2] = 3
    ff = defaultdict(lambda: float("inf"))
    ff[1] = 9
    ff[2] = 8
    ax, len1, len2 = visualize_frozen_updates([dd, ff], 2)
    assert ax.get_title() == "Timesteps before Updating Value"
    assert len1 == 1
    assert len2 == 2


def test_filter_inf_start_value():
    arr = [
        [float("inf"), 8, 9],
        [1, 2, 3],
        [float("inf"), 5, 7],
        [0, 0, 0],
    ]
    arr = np.array(arr)
    finite, infinite = filter_inf_start_value(arr)
    assert (
        infinite
        == np.array(
            [
                [float("inf"), 8.0, 9.0],
                [float("inf"), 5.0, 7.0],
            ]
        )
    ).all()
    assert (
        finite
        == np.array(
            [
                [1, 2, 3],
                [0, 0, 0],
            ]
        )
    ).all()


def test_plot_evals():
    df = pd.DataFrame(
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 2, 3],
        ]
    )
    ax = plot_evals(df)
    assert ax is not None
