import os
import logging
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.angles import convert_angle, angular_loss
from utils.device import get_device
from results import save_results

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


# Write the given line to the file of the provided name
def _write_to_file(filename, line):
    with open(filename, 'a') as file:
        file.write(line + '\n')


class GazeModel(nn.Module):
    def __init__(self, device=get_device()):
        super().__init__()
        self.name = f"GazeModel.pt"
        self.best_accuracy = None

        # Configure the device
        self.device = device
        self.to(device)

        # Optimizer and loss criteria
        self.optimizer = None
        self.l1_crit = nn.functional.l1_loss
        self.angular_crit = angular_loss

    def freeze_bn_layers(self):
        # Freeze all batch normalization layers
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def _train(self, train_data):
        self.train()

        avg_l1_loss, avg_angle_loss = 0, 0
        for data, label in tqdm(train_data):
            try:
                self.optimizer.zero_grad()
                label = label.to(self.device)
                output = self.forward(data)

                l1_loss = self.l1_crit(output, label)
                avg_l1_loss += l1_loss.item()
                avg_angle_loss += self.angular_crit(convert_angle(output), convert_angle(label)).item()

                # Update the network
                l1_loss.backward()
                self.optimizer.step()

            except Exception as e:
                # Log stack trace
                logging.error(e, exc_info=True)
                continue

        avg_l1_loss /= len(train_data)
        avg_angle_loss /= len(train_data)
        return avg_l1_loss, avg_angle_loss

    def _eval(self, validation_data):
        self.eval()

        avg_l1_loss, avg_angle_loss = 0, 0
        with torch.no_grad():
            for data, label in tqdm(validation_data):
                label = label.to(self.device)
                output = self.forward(data)

                avg_l1_loss += self.l1_crit(output, label).item()
                avg_angle_loss += self.angular_crit(convert_angle(output), convert_angle(label)).item()

        avg_l1_loss /= len(validation_data)
        avg_angle_loss /= len(validation_data)
        return avg_l1_loss, avg_angle_loss

    def learn(self, learn_data, validation_data, epochs, learning_rate, saves_dir, full_run, model_id):
        # Set the device and the optimizer
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize variables to track time and accuracies
        start_time = time.time()
        learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses = [], [], [], []

        for _ in tqdm(range(epochs)):
            learn_l1_loss, learn_angular_loss = self._train(learn_data)
            eval_l1_loss, eval_angular_loss = self._eval(validation_data)

            learn_l1_losses.append(learn_l1_loss)
            learn_angular_losses.append(learn_angular_loss)
            eval_l1_losses.append(eval_l1_loss)
            eval_angular_losses.append(eval_angular_loss)

            # Save the model when the new best evaluation loss is reached
            if self.best_accuracy is None or self.best_accuracy >= eval_angular_loss:
                self.best_accuracy = eval_angular_loss
                torch.save(self.state_dict(), os.path.join(saves_dir, self.name))

        # Get the total time spent
        total_time = time.time() - start_time

        # Save the losses over the epochs
        save_results(full_run, model_id, learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses)

        # Strip the model id of the report name and save the total time taken and the best accuracy achieved
        report_name = re.sub(rf'{model_id}\.pt$', '.txt', self.name)
        filename = f"reports/report{report_name}"
        task = "training" if re.search('[a-zA-Z]', model_id) is None else "calibration"
        _write_to_file(filename, f"{model_id} {task} results:\n    Total time taken: {total_time}"
                                 f"\n    Best accuracy achieved: {self.best_accuracy}")

    def inference(self, inference_data, model_id):
        # Set the device and eval state
        self.to(self.device)
        self.eval()

        # Run all the images for 2 images, one for warmup and one for real so there is no loading influence
        with torch.no_grad():
            for i in range(2):
                # Initialize variables to track time and accuracy for the actual run
                if i == 1:
                    start_time = time.time()
                    total_accuracy = 0

                for data, label in tqdm(inference_data):
                    label = label.to(self.device)
                    output = self.forward(data)
                    if i == 1:
                        total_accuracy += self.angular_crit(convert_angle(output), convert_angle(label)).item()

        # Get the images per second and the average accuracy over all the images
        total_time = time.time() - start_time
        images_per_second = len(inference_data) / total_time
        avg_accuracy = total_accuracy / len(inference_data)

        # Strip the model id of the report name and save the results
        report_name = re.sub(rf'{model_id}\.pt$', '.txt', self.name)
        filename = f"reports/report{report_name}"
        _write_to_file(filename, f"{model_id} inference results:\n    Images per second: {images_per_second}"
                                 f"\n    Average image accuracy: {avg_accuracy}")
