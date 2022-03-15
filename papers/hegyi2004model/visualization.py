import sys, os
sys.path.append(os.path.expanduser('~\\Documents\\git\\metanet'))
import metanet
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import datetime


def run_visualization(filename: str):
    # load simulation
    sim, other_data = metanet.io.load_sim(filename)

    # print data
    table = PrettyTable()
    table.field_names = ['Field', 'Value']
    table.float_format = '.3'
    
    exec_time = other_data.pop('execution_time', None)
    if exec_time is not None:
        table.add_row(
            ('execution time', datetime.timedelta(seconds=exec_time)))

    table.add_rows(list(other_data.items()))

    filename = f'From file: \'{filename}\''
    print(filename, table, sep='\n')

    # show plot
    fig, axs = sim.plot()
    fig.suptitle(filename, fontsize=14)
    # fig.savefig('test.eps')
    plt.show()


filename = (str(sys.argv[1]) 
            if len(sys.argv) > 1 else 
            'data/coord_opti.pkl')
run_visualization(filename)
