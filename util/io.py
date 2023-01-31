from logging import warn
from typing import Any, Collection, Iterable, Literal, Optional

import numpy as np
import numpy.typing as npt
from csnlp.util import io
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
    step_info_keys = processed_.keys()  # keys without "demands"
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
            warn(f"{nans} NaN detected during save (entry: '{k}').", RuntimeWarning)
        processed[k] = a
    return processed


def save_data(
    filename: str,
    agent_type: Literal["pk", "lstdq", "lstddpg"],
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
    agent_type : {"pk", "lstdq", "lstddpg"}
        Type of agent that was simulated.  Used to deduce which post-processing strategy
        to apply.
    data : collection
        Collection of simulation results.
    compression : {"lzma", "bz2", "gzip", "brotli", "blosc2", "matlab"]}
        Type of compression to apply to the file.

    info
        Any other piece of information to add to the file.
    """
    # process simulation data specifically for each agent
    env_data: Collection[MonitorInfos]
    # agent_data: list  # [RecordUpdates or something like that]
    if agent_type == "pk":
        env_data = data

    if agent_type in ("lstdq", "lstddpg"):
        # env_data, agent_data = zip(*data)
        raise NotImplementedError

    info["envs"] = postprocess_env_data(env_data)
    # info["agents"] = postprocess_agent_data(agent_data)
    io.save(filename, compression, **info)


load_data = io.load
