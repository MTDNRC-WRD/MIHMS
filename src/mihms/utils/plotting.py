import warnings

import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=DeprecationWarning)


def plot_stats(stats, file=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    stats = stats.loc['2017-01-01': '2017-12-31']
    ax.plot(stats['Date'], stats['basin_cfs_1'], color='r', linewidth=2.2, label="simulated")
    ax.plot(stats['Date'], stats['runoff_1'], color='b', linewidth=1.5, label="measured")
    ax.legend(bbox_to_anchor=(0.25, 0.65))
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow, in cfs")

    ax2 = ax.twinx()
    ax2.bar(stats['Date'], stats['basin_ppt_1'], label='precip', color='red', alpha=0.5, width=0.2)
    ax2.set_ylabel('precip', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    if file:
        plt.savefig(file)
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
