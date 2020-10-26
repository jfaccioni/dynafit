import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

PATH = ('/home/juliano/Dropbox/SHARED/DynaFit Experimental Cancer Cell/Submissions/'
        'Cancer Research Experimental/2nd Answer/Juliano/figures/experimental/')


def main(path: str) -> None:
    pal = sns.color_palette('rainbow', n_colors=20)
    c0 = pal[2]
    c1 = pal[15]
    c2 = pal[19]

    data0 = pd.read_excel(
        os.path.join(path, 'experimental_pop_growth_stats00.xlsx')
    )[['Static_GR_Final_N', 'Dynamic_GR_Final_N']]
    data1 = pd.read_excel(
        os.path.join(path, 'experimental_pop_growth_stats20.xlsx')
    )[['Static_GR_Final_N', 'Dynamic_GR_Final_N']]
    data2 = pd.read_excel(
        os.path.join(path, 'experimental_pop_growth_stats30.xlsx')
    )[['Static_GR_Final_N', 'Dynamic_GR_Final_N']]
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.scatter(data0.Static_GR_Final_N, data0.Dynamic_GR_Final_N, color=c0, s=30, alpha=0.7,
               label=r'Experimental $\sigma$')
    ax.scatter(data1.Static_GR_Final_N, data1.Dynamic_GR_Final_N, color=c1, s=30, alpha=0.7,
               label=r'Experimental $\sigma \times$ 2')
    ax.scatter(data2.Static_GR_Final_N, data2.Dynamic_GR_Final_N, color=c2, s=30, alpha=0.7,
               label=r'Experimental $\sigma \times$ 3')
    ax.scatter([1000], [1000], color='white', edgecolor='black', marker='*', s=200, label='start pos', zorder=2)
    ax.plot(ax.get_xlim(), ax.get_xlim(), lw=3, ls='--', color='k', alpha=0.7,
            scalex=False, scaley=False, zorder=0, label='$x = y$')
    fig.suptitle(r'Based on experimental data from Fulano $et \ al$ - 1000 initial cells' + '\n'
                 r'Growth Rate = $\frac{0.01215}{day}$ $\sigma$ = 0.02943')
    ax.set_xlabel('Final N - Fixed GR (mean of dynamic GR for the same replicate)')
    ax.set_ylabel('Final N - Dynamic GR (fluctuates on each generation)')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main(path=PATH)
