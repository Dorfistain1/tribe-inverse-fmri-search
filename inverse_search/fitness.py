from __future__ import annotations

from brain_utils import YEO7_NETWORKS, connectivity_matrix
from inverse_search.target import NeuralTarget
from tribe_core import BrainPrediction

_NETWORK_INDEX = {name: i for i, name in enumerate(YEO7_NETWORKS)}


def network_delta_score(prediction: BrainPrediction, target: NeuralTarget) -> float:
    """Default fitness function: for each (network_a, network_b) pair
    in the target, reward candidates whose predicted connectivity moves
    in the specified direction. Score is the mean of
    `direction * connectivity` across target pairs, so it's higher when
    "should increase" pairs are strongly connected and "should decrease"
    pairs are weakly (or negatively) connected.

    This scores absolute predicted connectivity, not a change relative
    to some baseline stimulus -- a real baseline-relative version
    requires a real measured contrast (see NeuralTarget's docstring on
    literature-encoded vs data-derived targets). Fine for a first
    end-to-end pipeline test; not a rigorous scientific claim yet.

    Requires prediction.activity to span multiple timesteps -- network
    connectivity is a correlation over time, not defined for one frame.
    """
    if prediction.activity.shape[0] < 2:
        raise ValueError(
            "network_delta_score needs a multi-timestep prediction "
            f"(got {prediction.activity.shape[0]} timestep) -- connectivity "
            "is a correlation over time."
        )
    conn = connectivity_matrix(prediction.activity)
    total = 0.0
    for (network_a, network_b), direction in target.network_deltas.items():
        total += direction * conn[_NETWORK_INDEX[network_a], _NETWORK_INDEX[network_b]]
    return total / len(target.network_deltas)
