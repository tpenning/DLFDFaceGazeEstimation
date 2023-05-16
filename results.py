import math
import argparse
from typing import List

import matplotlib.pyplot as plt


def save_results(fileid: str, train_l1_losses: List[float], train_angular_losses: List[float],
                 calibration_l1_losses: List[float], calibration_angular_losses: List[float],
                 eval1_l1_losses: List[float], eval1_angular_losses: List[float],
                 eval2_l1_losses: List[float], eval2_angular_losses: List[float]):
    # Generate the filename with the given id
    filename = f"results/result{fileid}.txt"

    # Write the results to the file
    with open(filename, "w") as file:
        for lst in [train_l1_losses, train_angular_losses, calibration_l1_losses, calibration_angular_losses,
                    eval1_l1_losses, eval1_angular_losses, eval2_l1_losses, eval2_angular_losses]:
            line = " ".join(str(item) for item in lst)
            file.write(line + "\n")


def _load_results(fileid: str):
    # Load the file by id
    filename = f"results/result{fileid}.txt"

    # Read the data from the file
    loaded_data = []
    with open(filename, "r") as file:
        for line in file:
            lst = [float(item) for item in line.strip().split()]
            loaded_data.append(lst)
    return loaded_data


def plot_results(fileid: str):
    # Load the results
    results = _load_results(fileid)

    # Plot titles and labels
    plot_titles = ["Training L1 Loss", "Training Angular Loss", "Calibration L1 Loss", "Calibration Angular Loss"]
    line_labels = ["Train", "Cal"]
    loss_metric_labels = ["L1 loss", "Angular loss"]

    # Create four subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    # Add the data for the plots
    for i in range(4):
        pi = format(i, '02b')
        plot = axs[int(pi[0]), int(pi[1])]

        plot.plot(results[i], "b-", label=line_labels[math.floor(i / 2)])
        plot.plot(results[i+4], "orange", label="Val")
        plot.set_title(plot_titles[i])
        plot.set_xlabel("Epoch")
        plot.set_ylabel(loss_metric_labels[i % 2])
        plot.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-fileid',
                        '--fileid',
                        type=int,
                        required=True,
                        help="id of the results file")

    args = parser.parse_args()
    plot_results(str(args.fileid))
