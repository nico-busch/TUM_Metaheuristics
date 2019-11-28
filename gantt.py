import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def create_gantt(prob, s, c):

    # create dataframe of schedule
    gates = ['Gate ' + str(x + 1) for x in s]
    flights = ['Flight ' + str(x + 1) for x in range(prob.n)]

    df = pd.DataFrame(data=zip(gates, prob.a, c, c + prob.d, prob.b, prob.d),
                      index=flights,
                      columns=['Gate', 'Earliest Start', 'Start', 'Finish', 'Latest Finish', 'Duration'])
    df = df.sort_values(['Gate', 'Start'])
    df.index.name = 'Flight'

    # plot gantt chart
    cmap = matplotlib.cm.get_cmap('Accent')
    col1 = col2 = cmap(range(cmap.N))
    col2[:, -1] = 0.75
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(18, 8))
    for i, gate in enumerate(df.groupby('Gate', sort=False)):
        for j, flight in enumerate(gate[1].groupby('Flight', sort=False)):
            plt.plot([flight[1]['Earliest Start'], flight[1]['Earliest Start']],
                     [i - 0.4, i + 0.4], color=col1[j], linewidth=2, zorder=0)
            plt.plot([flight[1]['Latest Finish'], flight[1]['Latest Finish']],
                     [i - 0.4, i + 0.4], color=col1[j], linewidth=2, zorder=0)
            data = flight[1][['Start', 'Duration']]
            chart = ax.broken_barh(data.values, (i - 0.25, 0.5), color=col2[j])
            plt.text((flight[1]['Start'] + flight[1]['Finish']) / 2, i, flight[0], ha='center', va='center')
    ax.set_yticks(range(prob.m))
    ax.set_yticklabels(['Gate ' + str(x + 1) for x in range(prob.m)][::-1])
    plt.tick_params(axis='y', which='both', left=False, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(xmin=0)

    plt.show()
