from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from util import load_data, parse_visualization_args, plot


def load_all_data(filenames: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Loads the data from the filenames and reduces their size."""
    for i, fn in enumerate(filenames):
        data = load_data(fn)
        name = Path(fn).stem
        details = (
            f" -{k}: {v}" for k, v in data.items() if k not in ("envs", "agents")
        )
        print(f"p{i}) {name}", *details, sep="\n")
        yield name, data


if __name__ == "__main__":
    # parse args and set default values for plotting
    args = parse_visualization_args()
    plot.set_mpl_defaults()

    # load and plot data one by one
    figs: dict[Callable[[Any], Figure], Figure] = {}
    for name, data in load_all_data(args.filenames):
        envsdata = data["envs"]
        agentsdata = data.get("agents", None)

        funcs: list[Callable[[Any], Figure]] = []
        if args.traffic:
            funcs.append(plot.plot_traffic_quantities)
        if args.cost:
            funcs.append(plot.plot_costs)
        if agentsdata is not None and args.agent:
            funcs.append(plot.plot_agent_quantities)

        for fun in funcs:
            figs[fun] = fun(  # type: ignore[call-arg]
                envsdata=envsdata,
                agentsdata=agentsdata,
                fig=figs.get(fun, None),
                label=name,
                reduce=args.reduce,
            )
    plt.show()
