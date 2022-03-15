import sys, os
sys.path.append(os.path.expanduser('~\\Documents\\git\\metanet'))
import metanet
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def run_visualization(filename: str):
    # load simulation
    sims, other_data = metanet.io.load_sims(filename)

    # due to a bug
    if isinstance(sims[0], list):
        sims = sims[0]

    # print data
    table = PrettyTable()
    table.field_names = ['Field', 'Value']
    table.float_format = '.3'
    table.add_rows(list(other_data.items()))

    txt = f'FILE: {filename}'
    print('', txt, table, sep='\n')

    # show plot of first sim
    fig, axs = sims[0].plot(sharex=True, drawstyle='steps-post', linestyle='-')
    for sim in sims[1:]:
        sim.plot(fig=fig, axs=axs, add_labels=False, linestyle='--',
                 drawstyle='steps-post')

    # add custom constraint
    axs[2, 1].axhline(y=other_data['w2_constraint'], linestyle='-.', 
                      linewidth=0.5, color='k')

    fig.suptitle(txt, fontsize=14)
    # fig.savefig('test.eps')
    plt.show()


filename = (str(sys.argv[1])
            if len(sys.argv) > 1 else
            'result.pkl')  # 'data/comparison6.pkl')

run_visualization(filename)
