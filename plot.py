from src.plots import collect_data, make_figure

data, metadata = collect_data(path="reports/cation_fraction")
x_data = [metadata[key]['mass fraction vaporized'] for key in list(sorted(metadata.keys()))]
y_data = {}
for i in list(sorted(data.keys())):
    d = data[i]
    for j in d.keys():
        if j not in y_data.keys():
            y_data.update({j: []})
        y_data[j].append(data[i][j])

make_figure(
    x_data=x_data,
    y_data=y_data,
    x_label="Mass Fraction Vaporized",
    y_label="System Cation Fraction"
)
