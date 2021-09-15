import os
import csv
import matplotlib.pyplot as plt

def collect_data(path, metadata_rows=2):
    data = {}
    metadata = {}
    iterations = list(sorted([i.split("/")[-1].replace(".csv", "") for i in os.listdir(path)]))
    for i in iterations:
        data.update({i: {}})
        metadata.update({i: {}})
        p = path + "/{}.csv".format(i)
        with open(p, 'r') as infile:
            reader = list(csv.reader(infile))
            for j in range(0, metadata_rows):
                line = reader[j]
                metadata[i].update({line[0]: float(line[1])})
            for j in range(metadata_rows, len(reader)):
                line = reader[j]
                data[i].update({line[0]: float(line[1])})
    return data, metadata



def make_figure(x_data, y_data, x_label, y_label):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for i in y_data.keys():
        ax.plot(
            x_data,
            y_data[i],
            linewidth=2.0,
            label=i
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    ax.legend()
    plt.show()
