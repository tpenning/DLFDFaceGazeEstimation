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
        self.name = "gaze-model.pt"
        self.device = device
        self.to(device)

        self.optimizer = None
        self.l1_crit = nn.functional.l1_loss
        self.angular_crit = angular_loss

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

    def _learn_step(self, learn_data, eval_data, epochs):
        learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses = [], [], [], []
        for _ in tqdm(range(epochs)):
            learn_l1_loss, learn_angular_loss = self._train(learn_data)
            eval_l1_loss, eval_angular_loss = self._eval(eval_data)

            learn_l1_losses.append(learn_l1_loss)
            learn_angular_losses.append(learn_angular_loss)
            eval_l1_losses.append(eval_l1_loss)
            eval_angular_losses.append(eval_angular_loss)

        return learn_l1_losses, learn_angular_losses, eval_l1_losses, eval_angular_losses

    def learn(self, train_data, calibration_data, validation_data, train_epochs, calibration_epochs,
              learning_rate, fileid):
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_l1_losses, train_angular_losses, eval1_l1_losses, eval1_angular_losses = \
            self._learn_step(train_data, validation_data, train_epochs)
        calibration_l1_losses, calibration_angular_losses, eval2_l1_losses, eval2_angular_losses = \
            self._learn_step(calibration_data, validation_data, calibration_epochs)

        save_results(fileid, train_l1_losses, train_angular_losses, calibration_l1_losses, calibration_angular_losses,
                     eval1_l1_losses, eval1_angular_losses, eval2_l1_losses, eval2_angular_losses)
