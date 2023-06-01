import numpy as np


# if __name__ == "__main__":
#     pids = [f"p{pid:02}" for pid in range(00, 15)]
#     averages_channel_Y = 4
#     averages_channel_Cb = 4
#     averages_channel_Cr = 4
#
#     for pid in pids:
#         # Load the images for a subject from the .npy file
#         person_images = np.load(f"../data/{pid}/images.npy")
#         person_averages_channel_Y = 4
#         person_averages_channel_Cb = 4
#         person_averages_channel_Cr = 4
#
#         averages_channel_Y += person_averages_channel_Y / len(person_images)
#         averages_channel_Cb += person_averages_channel_Cb / len(person_images)
#         averages_channel_Cr += person_averages_channel_Cr / len(person_images)
#
#     averages_channel_Y /= len(pids)
#     averages_channel_Cb /= len(pids)
#     averages_channel_Cr /= len(pids)
