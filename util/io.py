from argparse import Namespace
from logging import warn
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import scipy.io as spio
from mpcrl.wrappers.envs import MonitorInfos

_BAD_KEYS = ("__header__", "__version__", "__globals__")


def _check_keys(dict: Dict) -> Dict:
    """Checks if entries in dictionary are mat-objects. If yes, todict is called to
    change them to nested dictionaries."""
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj: spio.matlab.mio5_params.mat_struct) -> Dict:
    """A recursive function which constructs nested dictionaries from matobjects."""
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_data(filename: str) -> Dict[str, Any]:
    """Loads simulation data from .mat files.

    Many thanks to https://stackoverflow.com/a/8832212/19648688.
    """
    if not filename.endswith(".mat"):
        filename = f"{filename}.mat"
    data = _check_keys(spio.loadmat(filename, struct_as_record=False, squeeze_me=True))
    for k in _BAD_KEYS:
        data.pop(k, None)
    return data


def postprocess_env_data(
    data: List[MonitorInfos],
) -> Dict[str, npt.NDArray[np.floating]]:
    """Post-processes a list of monitored envs resulting from a simulation into a single
    dict of arrays.

    Parameters
    ----------
    data : List[MonitorInfos]
        List of monitored envs.

    Returns
    -------
    dict of (str, arrays)
        A dict of arrays.
    """
    # convert data into lists
    datum = data[0]
    processed_ = {k: [v] for k, v in datum.finalized_step_infos(np.nan).items()}
    processed_["demands"] = [datum.demands]
    for datum in data[1:]:
        # append step info
        info = datum.finalized_step_infos(np.nan)
        for k, v in processed_.items():
            v.append(info[k])

        # append demands
        processed_["demands"].append(datum.demands)

    # convert lists to arrays, adjust some shapes, and check for nans
    processed = {}
    for k, v in processed_.items():
        a = np.asarray(v)
        if k in ("state", "flow"):
            shape = a.shape
            a = a.reshape(*shape[:2], shape[2] * shape[3], shape[4])
        nans = np.isnan(a).sum()
        if nans > 0:
            warn(f"{nans} NaN detected during save (entry: '{k}').", RuntimeWarning)
        processed[k] = a
    return processed


def _save(filename: str, data: Dict[str, Any]) -> None:
    """Saves simulation data to a .mat file."""
    if not filename.endswith(".mat"):
        filename = f"{filename}.mat"
    spio.savemat(filename, data, do_compression=True, oned_as="column")


def save_data(
    filename: str,
    args: Namespace,
    data: List[Any],
    **others: Any,
) -> None:
    """Saves the simulation data to a .mat file.

    Parameters
    ----------
    filename : str
        Destination filename. If the extension `'.mat'` is missing, it is automatically
        added.
    args : Namespace
        Arguments used to run the simulation to save. Also used to deduce which
        post-processing to apply to the data based on the agent's type.
    data : List[Any]
        List of simulation results.
    others
        Any other piece of information to add to the .mat file.
    """
    # add things in common to the dict
    mdict = others
    mdict["args"] = args.__dict__

    # process simulation data specifically for each agent
    env_data: List[MonitorInfos]
    # agent_data: List  # [RecordUpdates or something like that]
    if args.agent_type == "pk":
        env_data = data

    if args.agent_type in ("lstdq", "lstddpg"):
        # env_data, agent_data = zip(*data)
        raise NotImplementedError

    mdict["envs"] = postprocess_env_data(env_data)
    # mdict["agents"] = postprocess_agent_data(agent_data)
    _save(filename, mdict)
