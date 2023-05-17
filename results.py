import math
import argparse
import re
from typing import List

import matplotlib.pyplot as plt


def save_results(model_id: str, learn_l1_losses: List[float], learn_angular_losses: List[float],
                 eval_l1_losses: List[float], eval_angular_losses: List[float]):
    # Get the losses in a list, when testing also get the training losses from file
    if model_id.startswith("Train"):
        losses = [learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses]
    else:
        train_id = re.split("(\D+)", model_id)[0]
        train_losses = _load_results(f"Train{train_id}")
        losses = [train_losses[0], train_losses[1], learn_l1_losses, learn_angular_losses,
                  train_losses[2], train_losses[3], eval_l1_losses, eval_angular_losses]

    # Generate the filename with the given id
    filename = f"results/result{model_id}.txt"

    # Write the results to the file
    with open(filename, "w") as file:
        for lst in losses:
            line = " ".join(str(item) for item in lst)
            file.write(line + "\n")


def _load_results(results_id: str):
    # Load the file by id
    filename = f"results/result{results_id}.txt"

    # Read the data from the file
    loaded_data = []
    with open(filename, "r") as file:
        for line in file:
            lst = [float(item) for item in line.strip().split()]
            loaded_data.append(lst)
    return loaded_data


def plot_results(result_id: str):
    # Check if this is just training data or training and calibration
    full_plot = not result_id.startswith("Train")

    # Load the results
    results = _load_results(result_id)

    # Plot titles and labels
    plot_titles = ["Training L1 Loss", "Training Angular Loss", "Calibration L1 Loss", "Calibration Angular Loss"]
    line_labels = ["Train", "Cal"]
    loss_metric_labels = ["L1 loss", "Angular loss"]

    # Create four subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6)) if full_plot else \
        plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    # Add the data for the plots
    plot_count = 4 if full_plot else 2
    for i in range(plot_count):
        pi = format(i, '02b')
        plot = axs[int(pi[0]), int(pi[1])] if full_plot else axs[int(pi[1])]

        plot.plot(results[i], "b-", label=line_labels[math.floor(i / 2)])
        plot.plot(results[i + plot_count], "orange", label="Val")
        plot.set_title(plot_titles[i])
        plot.set_xlabel("Epoch")
        plot.set_ylabel(loss_metric_labels[i % 2])
        plot.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-result_id',
                        '--result_id',
                        type=str,
                        required=True,
                        help="id of the results")

    args = parser.parse_args()
    plot_results(args.result_id)
