import matplotlib.pyplot as plt

images_per_second_data = [771.881577141754, 541.263055762412, 366.0787020519901, 307.18302359713323, 664.7136769290025, 491.49882117432907, 341.18385973661634, 290.85147112007604, 772.7076497282071, 773.0294903364604, 374.22460211839126, 322.4585363882076, 1078.6265516848648, 970.641655420759, 486.7352725096087, 368.8284771460681, 1050.738110560303, 959.9800615647994, 482.7620665974614, 366.4389999243811, 1037.2169400117057, 959.968481894052, 486.90109275940824, 367.3458144207706, 1090.3369508883768, 963.7459501922201, 495.81148096667584, 368.64843778494804, 1034.9910316961564, 930.226350646414, 448.2942866791236, 347.3616749982134, 972.9064235317061, 734.1450723558112, 457.9373583330809, 330.77933064991464, 567.2376470873419, 481.51061535425487, 318.77465397008746, 257.76083328236183]
errors_data = [3.291496332802872, 3.3040587085731326, 2.8197992190447945, 2.7167484778791664, 3.315541807938864, 3.2603707143383716, 2.7568123208578674, 2.6510331710283337, 3.500361073578646, 3.4215902414019532, 3.3474976876143367, 3.135506631353249, 3.686051917564061, 3.5748863305967298, 3.4551080306337525, 3.208804302547996, 3.4084280825352913, 3.426027815249811, 3.336437242256229, 3.128667013182553, 3.366571943690007, 3.3107775027432167, 3.441402658065781, 3.142568589543365, 3.4228922505926342, 3.357660630505097, 3.264161590142796, 3.044219057594426, 3.388894560673026, 3.3717243688811855, 3.3699224577737974, 3.1065332311386245, 3.3837034902329868, 3.364417593378574, 3.361651863812841, 3.070501480595457, 3.6343616403490926, 3.476562261668841, 3.3853712027829763, 3.2692418769265834]


if __name__ == "__main__":
    # Data
    architectures = ["AlexNet LC", "AlexNet HC", "ResNet-18 LC", "ResNet-18 HC"]
    data_types = ["RGB", "YCbCr", "FD0", "FD1", "FD2", "FD3", "FD4", "FD5", "FD6", "FDD7"]

    # Inference results
    images_per_second = images_per_second_data
    errors = errors_data

    # Define colors and shapes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    shapes = ['o', 's', 'v', '^']

    # Create the plot
    plt.figure(figsize=(14, 6))

    for i, data_type in enumerate(data_types):
        for j, architecture in enumerate(architectures):
            idx = i * len(architectures) + j
            plt.scatter(images_per_second[idx], errors[idx], color=colors[i], marker=shapes[j], s=70, label=None)

    # Set axis labels and title
    plt.xlabel("Images per Second")
    plt.ylabel("Error (degrees)")
    # plt.title("Inference Results")

    # Combine legends for colors and shapes into a single legend
    legend_elements = []
    for i, data_type in enumerate(data_types):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10))
    legend_elements.extend([plt.Line2D([0], [0], marker=shape, color='black', linestyle='None', markersize=10) for shape in shapes])

    # Set the combined legend
    plt.legend(legend_elements, data_types + architectures, title="Data Types and Architectures", loc='lower right')

    # Show the plot
    plt.tight_layout()
    plt.show()
