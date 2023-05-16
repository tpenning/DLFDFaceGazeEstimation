import math
from typing import List
from datetime import datetime

import matplotlib.pyplot as plt


def save_results(train_l1_losses: List[float], train_angular_losses: List[float],
                 eval1_l1_losses: List[float], eval1_angular_losses: List[float],
                 calibration_l1_losses: List[float], calibration_angular_losses: List[float],
                 eval2_l1_losses: List[float], eval2_angular_losses: List[float]):
    # Generate the filename based on the time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"results_{current_time}.txt"

    # Write the results to the file
    with open(filename, "w") as file:
        for lst in [train_l1_losses, train_angular_losses, eval1_l1_losses, eval1_angular_losses,
                    calibration_l1_losses, calibration_angular_losses, eval2_l1_losses, eval2_angular_losses]:
            line = " ".join(str(item) for item in lst)
            file.write(line + "\n")


def _load_results(filename: str):
    # Read the data from the file
    loaded_data = []
    with open(filename, "r") as file:
        for line in file:
            lst = [int(item) for item in line.strip().split()]
            loaded_data.append(lst)
    return loaded_data


def plot_results(filename: str):
    # Load the results
    results = _load_results(filename)

    # Plot titles and labels
    plot_titles = ["Training L1 Loss", "Training Angular Loss", "Calibration L1 Loss", "Calibration Angular Loss"]
    line_labels = ["Train", "Cal"]
    loss_metric_labels = ["L1 loss", "Angular loss"]

    # Create four subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Add the data for the plots
    for i in range(4):
        pi = format(i, '02b')
        plot = axs[int(pi[0]), int(pi[1])]
        data_index = i * 2

        plot.plot(results[data_index], "b-", label=line_labels[math.floor(i / 2)])
        plot.plot(results[data_index + 1], "orange", label="Val")
        plot.set_title(plot_titles[i])
        plot.set_xlabel("Epoch")
        plot.set_ylabel(loss_metric_labels[i % 2])
        plot.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()
