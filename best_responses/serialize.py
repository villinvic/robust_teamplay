import os
import yaml
import numpy as np

from constants import Paths


def load_best_response_utilities(env_name: str) -> dict:
    path = Paths.BEST_RESPONSES.format(env=env_name)
    best_response_utilities = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            best_response_utilities = yaml.safe_load(f)

    return best_response_utilities


def save_best_response_utilities(env_name: str, best_response_utilities: dict):
    path = Paths.BEST_RESPONSES.format(env=env_name)

    to_save = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            best_response_utilities = yaml.safe_load(f)
            if best_response_utilities is None:
                best_response_utilities = {}
            to_save.update(best_response_utilities)

    # Update the best responses with the better ones found here.
    for scenario, new_value in best_response_utilities.items():
        if to_save.get(scenario, -np.inf) < new_value:
            to_save[scenario] = new_value

    with open(path, "w") as f:
        yaml.safe_dump(to_save, f)
