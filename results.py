import math
import argparse
import re
from typing import List

import matplotlib.pyplot as plt


def save_results(full_run, model_id: str, test_id: str, learn_l1_losses: List[float], learn_angular_losses: List[float],
                 eval_l1_losses: List[float], eval_angular_losses: List[float]):
    # Get the losses in a list, when testing also get the training losses from file
    # And generate the filename with the given id
    if full_run:
        train_losses = _load_results(False, re.split("(\D+)", model_id)[0], test_id)
        losses = [train_losses[0], train_losses[1], learn_l1_losses, learn_angular_losses,
                  train_losses[2], train_losses[3], eval_l1_losses, eval_angular_losses]
        prefix = "Test"
    else:
        losses = [learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses]
        prefix = "Train"
    filename = f"results/result{prefix}{model_id}_{test_id}.txt"

    # Write the results to the file
    with open(filename, "w") as file:
        for lst in losses:
            line = " ".join(str(item) for item in lst)
            file.write(line + "\n")


def _load_results(full_run: bool, result_id: str, test_id: str):
    # Load the file by id
    prefix = "Test" if full_run else "Train"
    filename = f"results/result{prefix}{result_id}_{test_id}.txt"

    # Read the data from the file
    loaded_data = []
    with open(filename, "r") as file:
        for line in file:
            lst = [float(item) for item in line.strip().split()]
            loaded_data.append(lst)

    return loaded_data


def plot_results(args):
    # Defines if only training or a full plot has to be loaded and plotted
    full_plot = args.full_run == "full"

    # Fix the test_id and load the results
    test_id = str(args.test_id).zfill(2)
    results = _load_results(full_plot, args.result_id, test_id)

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

    parser.add_argument('-full_run',
                        '--full_run',
                        default="full",
                        type=str,
                        required=False,
                        help="whether just the full run has to be plotted or not")

    parser.add_argument('-result_id',
                        '--result_id',
                        type=str,
                        required=True,
                        help="id of the results")

    parser.add_argument('-test_id',
                        '--test_id',
                        default=14,
                        type=int,
                        required=False,
                        help="id of the subject used for testing")

    args = parser.parse_args()
    plot_results(args)
