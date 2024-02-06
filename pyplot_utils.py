from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def set_box_color(bp):
    color = "#555555"
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['means'], color=color)


def make_grouped_boxplot(data, name="grouped_boxplot", whiskers=(0, 100)):
    # data[approach][metric (run type)]

    plt.figure()

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'lightcoral']

    num_runs = len(data)

    widths = 1. / (num_runs * 0.9)


    spacings = np.linspace(-0.7 + widths * 0.5, 0.7 - widths * 0.5, num=num_runs)

    datamin = np.inf
    datamax = -np.inf
    for run, color, label, spacing in zip(data.values(), colors, data.keys(), spacings):

        data = np.array(list(run.values()), dtype=object)
        datamin = min([v for sublist in data for v in sublist] + [datamin])
        datamax = max([v for sublist in data for v in sublist] + [datamax])

        boxplot = plt.boxplot(
            data, positions=np.arange(len(data))*2.0+spacing, sym='', widths=widths,
            whis=whiskers,
            patch_artist=True,
            meanline=True, showmeans=True,
        )

        set_box_color(boxplot)
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color=color)

        plt.plot([], c=color, label=label)

    plt.legend()

    plt.xticks(range(0, len(run) * 2, 2), run.keys(), rotation=45)
    plt.xlim(-2, len(run)*2)

    norm = (datamax - datamin) * 0.1

    plt.ylim(np.maximum(datamin - norm, -1e-3), datamax + norm * 3)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{name}.png')

    plt.clf()


def make_grouped_plot(data, name):
    plt.figure()

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'lightcoral'] + [
        tuple(np.random.random(3)) for _ in range(100)
    ]
    markers = ['o', 's', '^', '*', 'v'] + [''] * 1000

    data_per_run = defaultdict(dict)

    for approach, metrics in data.items():
        for metric, values in metrics.items():
            data_per_run[metric][approach] = values

    for (metric, approaches) in data_per_run.items():

        for (approach, values), color, marker in zip(approaches.items(),  colors, markers):

            plt.plot(values, label=approach, color=color)


        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel(metric)
        plt.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{name}_{metric}.png')
        print("saved ", f'{name}_{metric}.png')
        plt.clf()


if __name__ == '__main__':

    np.random.seed(0)
    data  = {
        f"approach_{approach}": {
            f"run_type_{run_type}": np.random.random(10) for run_type in range(4)
        }
        for approach in range(3)
    }


    make_grouped_boxplot(data, name="test")