import os
import logging
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


class GazeModel(nn.Module):
    def __init__(self, device=get_device()):
        super().__init__()
        self.name = f"GazeModel.pt"

        # Configure the device
        self.device = device
        self.to(device)

        # Optimizer and loss criteria
        self.optimizer = None
        self.l1_crit = nn.functional.l1_loss
        self.angular_crit = angular_loss

    def freeze_bn_layers(self):
        # Freeze all Batch Normalization (BN) layers
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

        # Train and evaluate with the given datasets
        learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses = [], [], [], []
        for _ in tqdm(range(epochs)):
            learn_l1_loss, learn_angular_loss = self._train(learn_data)
            eval_l1_loss, eval_angular_loss = self._eval(validation_data)

            learn_l1_losses.append(learn_l1_loss)
            learn_angular_losses.append(learn_angular_loss)
            eval_l1_losses.append(eval_l1_loss)
            eval_angular_losses.append(eval_angular_loss)

        # Save the model and the losses
        torch.save(self.state_dict(), os.path.join(saves_dir, self.name))
        save_results(full_run, model_id, learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses)
