import matplotlib.pyplot as plt
from src.plots import collect_data

runs = {
    "Canonical": {},
    "Half-Earths": {},
}

for run in runs.keys():
    runs[run] = collect_data(run)
