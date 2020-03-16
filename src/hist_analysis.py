import matplotlib.pyplot as plt
import numpy as np

from src.random_sampling import add_bins, load_data

SETTINGS = {
    # path to input file.
    'path': '../data/Pasta para Ju.xlsx',
    # Colony Size cell range in input Excel file
    'cs_range': 'A4:A1071',
    # Growth Rate range in input Excel file
    'gr_range': 'B4:B1071',
    # max size of colony to not be binned, e.g. a parameter of 10 makes colonies of up to 10 cells to be
    # plotted individually in CVP, instead of being binned along with other cell sizes.
    'max_binned_colony_size': 10,
    # number of bins in which to divide the population after the max number of individual colony sizes
    # determined by the parameter before.
    'bins': 10
}


def main(path: str, cs_range: str, gr_range: str, max_binned_colony_size: int, bins: int) -> None:
    """Main function of this script"""
    data = load_data(path=path, cs_range=cs_range, gr_range=gr_range)
    data = add_bins(data=data, max_binned_colony_size=max_binned_colony_size, bins=bins)
    grouped_data = data.groupby('bins')
    fig, ax = plt.subplots()
    ax.hist(np.log2(data['CS1']))
    ax.set_xlabel('log2(colony size)')
    ax.set_ylabel('count')
    ax.set_title('Colony size histogram')
    for x, label in zip(np.log2(grouped_data.max()['CS1']), grouped_data.count()['CS1']):
        ax.axvline(x, c='k')
        ax.text(x, ax.get_ylim()[1]*0.9, label)
    plt.show()


if __name__ == '__main__':
    main(**SETTINGS)
