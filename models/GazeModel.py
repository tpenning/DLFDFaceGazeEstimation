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
from utils.write_help import write_to_file
from results import save_results

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class GazeModel(nn.Module):
    def __init__(self, model_name: str, experiment: str, channel_regularization: float, dynamic=False, device=get_device()):
        super().__init__()
        # Configure the model
        self.name = model_name
        self.experiment = experiment == "experiment"
        self.channel_regularization = channel_regularization
        self.dynamic = dynamic
        self.best_accuracy = None
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

    def l1_cs_crit(self, output, label):
        if self.dynamic:
            # Determine the loss based on l1 and possible the channel amount for FDD
            l1_loss = self.l1_crit(output[:, :2], label)
            cs_loss = self.channel_regularization * torch.mean(output[:, 2])
            l1_cs_loss = l1_loss + cs_loss

            return l1_cs_loss
        else:
            # Use just l1 for the loss
            return self.l1_crit(output, label)

    def _train(self, train_data):
        self.train()

        avg_l1_loss, avg_angle_loss = 0, 0
        for data, label in tqdm(train_data):
            try:
                self.optimizer.zero_grad()
                label = label.to(self.device)
                output = self.forward(data)

                # The dynamic model gumbel softmax can rarely create a nan which has to be removed (seems cpu only)
                # Checking continuously would be too expensive, therefore the entire batch is excluded
                if torch.isnan(output).any().item():
                    raise ValueError("A nan value was created in the gumbel softmax of the dynamic model layers. "
                                     "This batch is hereby excluded.")

                # Correct the output for FDD selected channel amount
                if self.dynamic:
                    angle = output[:, :2]
                else:
                    angle = output

                l1_cs_loss = self.l1_cs_crit(output, label)
                avg_l1_loss += l1_cs_loss.item()
                avg_angle_loss += self.angular_crit(convert_angle(angle), convert_angle(label)).item()

                # Update the network
                l1_cs_loss.backward()
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
                try:
                    label = label.to(self.device)
                    output = self.forward(data)

                    # The dynamic model gumbel softmax can rarely create a nan which has to be removed (seems cpu only)
                    # Checking continuously would be too expensive, therefore the entire batch is excluded
                    if torch.isnan(output).any().item():
                        raise ValueError("A nan value was created in the gumbel softmax of the dynamic model layers. "
                                         "This batch is hereby excluded.")

                    # Correct the output for FDD selected channel amount
                    if self.dynamic:
                        angle = output[:, :2]
                    else:
                        angle = output

                    l1_cs_loss = self.l1_cs_crit(output, label)
                    avg_l1_loss += l1_cs_loss.item()
                    avg_angle_loss += self.angular_crit(convert_angle(angle), convert_angle(label)).item()

                except Exception as e:
                    # Log stack trace
                    logging.error(e, exc_info=True)
                    continue

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
        task = "training" if re.search('[a-zA-Z]', model_id) is None else "calibration"
        newline = "" if self.experiment else "\n"
        experiment_run = "Experiment" if self.experiment else ""
        report_name = re.sub(rf'{model_id}\.pt$', '.txt', self.name)

        filename = f"reports/report{experiment_run}{report_name}"
        write_to_file(filename, f"{model_id} {task} time: {total_time}")
        write_to_file(filename, f"{model_id} {task} accuracy: {self.best_accuracy}{newline}")

    def inference(self, inference_data, model_id):
        # Set the device and eval state
        self.to(self.device)
        self.eval()

        total_images = len(inference_data)
        # Run all the images for 2 images, one for warmup and one for real so there is no loading influence
        with torch.no_grad():
            for i in range(2):
                # Initialize variables to track time and accuracy for the actual run
                if i == 1:
                    start_time = time.time()
                    total_accuracy = 0

                for data, label in tqdm(inference_data):
                    try:
                        label = label.to(self.device)
                        output = self.forward(data)

                        # The dynamic model gumbel softmax can rarely create a nan which has to be removed (seems cpu only)
                        # Checking continuously would be too expensive, therefore the entire batch is excluded
                        if torch.isnan(output).any().item():
                            total_images -= 1
                            raise ValueError("A nan value was created in the gumbel softmax of the dynamic model layers. "
                                             "This batch is hereby excluded.")

                        # Correct the output for FDD selected channel amount
                        if self.dynamic:
                            angle = output[:, :2]
                        else:
                            angle = output

                        if i == 1:
                            total_accuracy += self.angular_crit(convert_angle(angle), convert_angle(label)).item()

                    except Exception as e:
                        # Log stack trace
                        logging.error(e, exc_info=True)
                        continue

        # Get the images per second and the average accuracy over all the images
        total_time = time.time() - start_time
        images_per_second = total_images / total_time
        avg_accuracy = total_accuracy / total_images

        # Strip the model id of the report name and save the results
        experiment_run = "Experiment" if self.experiment else ""
        report_name = re.sub(rf'{model_id}\.pt$', '.txt', self.name)
        filename = f"reports/report{experiment_run}{report_name}"
        write_to_file(filename, f"{model_id} inference time: {images_per_second}")
        write_to_file(filename, f"{model_id} inference accuracy: {avg_accuracy}\n")
