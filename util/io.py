from typing import Any, Collection, Iterable, Literal, Optional
from warnings import warn

import numpy as np
import numpy.typing as npt
from csnlp.util import io
from mpcrl.wrappers.agents import RecordUpdates
from mpcrl.wrappers.envs import MonitorInfos


def postprocess_env_data(
    data: Iterable[MonitorInfos],
) -> dict[str, npt.NDArray[np.floating]]:
    """Post-processes a list of monitored envs resulting from a simulation into a single
    dict of arrays.

    Parameters
    ----------
    data : Iterable[MonitorInfos]
        An iterable of monitored envs.

    Returns
    -------
    dict of (str, arrays)
        A dict of arrays.
    """
    # convert data into lists
    dataiter = iter(data)
    datum = next(dataiter)
    processed_ = {k: [v] for k, v in datum.finalized_step_infos(np.nan).items()}
    step_info_keys = list(processed_.keys())  # keys without "demands"
    processed_["demands"] = [datum.demands]
    for datum in dataiter:
        # append step info
        info = datum.finalized_step_infos(np.nan)
        for k in step_info_keys:
            processed_[k].append(info[k])

        # append demands
        processed_["demands"].append(datum.demands)

    # convert lists to arrays, adjust some shapes, and check for nans
    processed = {}
    for k, v in processed_.items():
        a = np.asarray(v)
        nans = np.isnan(a).sum()
        if nans > 0:
            warn(f"{nans} NaN detected in env save (entry: '{k}').", RuntimeWarning)
        processed[k] = a
    return processed


def postprocess_agent_data(
    agent_type: Literal["lstdq"],
    data: Iterable[RecordUpdates],
) -> dict[str, npt.NDArray[np.floating]]:
    """Post-processes a list of agents resulting from a simulation into a single dict of
    arrays.

    Parameters
    ----------
    data : Iterable[MonitorInfos]
        An iterable of learning-based agents.

    Returns
    -------
    dict of (str, arrays)
        A dict of arrays.
    """
    # convert data into lists
    dataiter = iter(data)
    datum = next(dataiter)
    processed_ = {k: [v] for k, v in datum.updates_history.items()}
    parnames = list(processed_.keys())  # keys without special quantities

    # save special quantities
    if agent_type == "lstdq":
        processed_["td_errors"] = [datum.td_errors]

    for datum in dataiter:
        # append updates history
        history = datum.updates_history
        for k in parnames:
            processed_[k].append(history[k])

        # save special quantities
        if agent_type == "lstdq":
            processed_["td_errors"].append(datum.td_errors)

    # convert lists to arrays, adjust some shapes, and check for nans
    processed = {}
    for k, v in processed_.items():
        a = np.asarray(v)
        nans = np.isnan(a).sum()
        if nans > 0:
            warn(f"{nans} NaN detected in agent save (entry: '{k}').", RuntimeWarning)
        processed[k] = a
    return processed


def save_data(
    filename: str,
    agent_type: Literal["pk", "lstdq"],
    data: Collection[Any],
    compression: Optional[
        Literal["lzma", "bz2", "gzip", "brotli", "blosc2", "matlab"]
    ] = None,
    **info: Any,
) -> None:
    """Saves the simulation data to a file.

    Parameters
    ----------
    filename : str
        The name of the file to save to. If the filename does not end in the correct
        extension, then it is automatically added. The extensions are
         - "pickle": .pkl
         - "lzma": .xz
         - "bz2": .pbz2
         - "gzip": .gz
         - "brotli": .bt
         - "blosc2": .bl2
         - "matlab": .mat
    agent_type : {"pk", "lstdq"}
        Type of agent that was simulated.  Used to deduce which post-processing strategy
        to apply.
    data : collection
        Collection of simulation results, either a list of envs or (envs, agents).
    compression : {"lzma", "bz2", "gzip", "brotli", "blosc2", "matlab"]}
        Type of compression to apply to the file.

    info
        Any other piece of information to add to the file.
    """
    if agent_type == "pk":
        info["envs"] = postprocess_env_data(data)
    elif agent_type == "lstdq":
        env_data, agent_data = zip(*data)
        info["envs"] = postprocess_env_data(env_data)
        info["agents"] = postprocess_agent_data(agent_type, agent_data)
    io.save(filename, compression, **info)


load_data = io.load
