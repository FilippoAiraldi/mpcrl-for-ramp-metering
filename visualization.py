from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from util import load_data, parse_visualization_args, plot


def load_all_data(
    filenames: Iterable[str],
) -> Iterator[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Loads the data from the filenames and reduces their size."""
    for i, fn in enumerate(filenames):
        name = Path(fn).stem
        data = load_data(fn)
        envsdata = data.pop("envs")
        agentsdata = data.pop("agents", {})
        details = (f" -{k}: {v}" for k, v in data.items())
        print(f"p{i}) {name}", *details, sep="\n")
        yield name, envsdata, agentsdata


if __name__ == "__main__":
    # parse args and set default values for plotting
    args = parse_visualization_args()
    plot.set_mpl_defaults()

    # load all data and print simulation details
    names, envsdata, agentsdata = zip(*load_all_data(args.filenames))

    # plot each figure at a time
    if args.traffic:
        plot.plot_traffic_quantities(envsdata, names, reduce=args.reduce)
    if args.cost:
        plot.plot_costs(envsdata, names)
    if args.agent:
        plot.plot_agent_quantities(agentsdata, names, reduce=args.reduce)
    plt.show()
