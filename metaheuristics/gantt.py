from itertools import cycle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

'''
    This file contains a function to create a gantt chart with the inputs: 
        - prob = problem instance
        - s = (feasible) sequence of flight to gate allocations (index = flight number, value = allocated gate)
        - c = (feasible) sequence of starting times for gate occupation for each flight (index = flight number,
               value = scheduled time)   
'''


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
    cmap = matplotlib.cm.get_cmap('tab20')
    cmap_colors = cmap(range(cmap.N))
    cmap_cycle = cycle(cmap_colors)
    colors = []
    for i in flights:
        colors.append(next(cmap_cycle))

    # plot gantt chart
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(18, 8))
    for i, gate in enumerate(df.groupby('Gate', sort=False)):
        switch = True
        for j, flight in enumerate(gate[1].groupby('Flight', sort=False)):
            y = 0 if switch else 0.2
            plt.plot([flight[1]['Earliest Start'].iloc[0], flight[1]['Earliest Start'].iloc[0]],
                     [gate[0] - 0.2 - y, gate[0] + 0.4 - y], color=colors[flight[0] - 1], linewidth=1, zorder=0)
            plt.plot([flight[1]['Latest Finish'].iloc[0], flight[1]['Latest Finish'].iloc[0]],
                     [gate[0] - 0.2 - y, gate[0] + 0.4 - y], color=colors[flight[0] - 1], linewidth=1, zorder=0)
            plt.arrow(flight[1]['Earliest Start'].iloc[0], gate[0] + 0.4 - 4 * y,
                      flight[1]['Latest Finish'].iloc[0] - flight[1]['Earliest Start'].iloc[0], 0,
                      head_width=0.05, head_length=1, length_includes_head=True, color=colors[flight[0] - 1])
            plt.arrow(flight[1]['Latest Finish'].iloc[0], gate[0] + 0.4 - 4 * y,
                      flight[1]['Earliest Start'].iloc[0] - flight[1]['Latest Finish'].iloc[0], 0,
                      head_width=0.05, head_length=1, length_includes_head=True, color=colors[flight[0] - 1])
            switch = not switch
            data = flight[1][['Start', 'Duration']]
            ax.broken_barh(data.values, (gate[0] - 0.2, 0.4), color=colors[flight[0] - 1])
            plt.text((flight[1]['Start'] + flight[1]['Finish']) / 2, gate[0],
                     'Flight ' + str(flight[0]), ha='center', va='center')
    ax.set_yticks(range(1, prob.m + 1))
    ax.set_yticklabels(['Gate ' + str(x + 1) for x in range(prob.m)])
    plt.tick_params(axis='y', which='both', left=False, pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(xmin=0)
    plt.show()
