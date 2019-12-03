from itertools import cycle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def create_gantt(prob, s, c):

    # create dataframe of schedule
    gates = s + 1
    flights = range(1, prob.n + 1)

    df = pd.DataFrame(data=zip(gates, prob.a, c, c + prob.d, prob.b, prob.d),
                      index=flights,
                      columns=['Gate', 'Earliest Start', 'Start', 'Finish', 'Latest Finish', 'Duration'])
    df = df.sort_values(['Gate', 'Start'])
    df.index.name = 'Flight'

    # create colors
    cmap = matplotlib.cm.get_cmap('Dark2')
    cmap_colors = cmap(range(cmap.N))
    cmap_colors_t = cmap_colors
    cmap_colors_t[:, -1] = 0.75
    cmap_cycle = cycle(cmap_colors)
    cmap_cycle_t = cycle(cmap_colors_t)
    colors, colors_t = [], []
    for i in flights:
        colors.append(next(cmap_cycle))
        colors_t.append(next(cmap_cycle_t))

    # plot gantt chart
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(18, 8))
    for i, gate in enumerate(df.groupby('Gate', sort=False)):
        for j, flight in enumerate(gate[1].groupby('Flight', sort=False)):
            print(flight[1]['Earliest Start'])
            print(tuple(colors[flight[0]]))
            print(flight[1])
            plt.plot([flight[1]['Earliest Start'], flight[1]['Earliest Start']],
                     [i - 0.4, i + 0.4], color=tuple(colors[flight[0]]), linewidth=2, zorder=0)
            plt.plot([flight[1]['Latest Finish'], flight[1]['Latest Finish']],
                     [i - 0.4, i + 0.4], color=tuple(colors[flight[0]]), linewidth=2, zorder=0)
            data = flight[1][['Start', 'Duration']]
            chart = ax.broken_barh(data.values, (i - 0.25, 0.5), color=tuple(colors_t[flight[0]]))
            plt.text((flight[1]['Start'] + flight[1]['Finish']) / 2, i, flight[0], ha='center', va='center')
    ax.set_yticks(range(prob.m))
    ax.set_yticklabels(['Gate ' + str(x + 1) for x in range(prob.m)][::-1])
    plt.tick_params(axis='y', which='both', left=False, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(xmin=0)

    plt.show()
