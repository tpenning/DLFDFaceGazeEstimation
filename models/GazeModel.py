import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.angles import convert_angle, avg_angle_diff
from utils.device import get_device

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class GazeModel(nn.Module):
    def __init__(self, device=get_device()):
        super().__init__()
        self.name = "gaze-model.pt"
        self.device = device
        self.to(device)

        self.train_data = None
        self.eval_data = None
        self.optimizer = None
        self.l1_crit = nn.functional.l1_loss
        self.deg_crit = avg_angle_diff

    def _train(self):
        self.train()

        avg_l1_loss, avg_angle_loss = 0, 0
        for data, label in tqdm(self.train_data):
            try:
                self.optimizer.zero_grad()
                label = label.to(self.device)
                output = self.forward(data)

                l1_loss = self.l1_crit(output, label)
                avg_l1_loss += l1_loss.item()
                avg_angle_loss += self.deg_crit(convert_angle(output), convert_angle(label)).item()

                # Update the network
                l1_loss.backward()
                self.optimizer.step()

            except Exception as e:
                # Log stack trace
                logging.error(e, exc_info=True)
                continue

        avg_l1_loss /= len(self.train_data)
        avg_angle_loss /= len(self.train_data)
        return avg_l1_loss, avg_angle_loss

    def _eval(self):
        self.eval()

        avg_l1_loss, avg_angle_loss = 0, 0
        with torch.no_grad():
            for data, label in tqdm(self.eval_data):
                label = label.to(self.device)
                output = self.forward(data)

                avg_l1_loss += self.l1_crit(output, label).item()
                avg_angle_loss += self.deg_crit(convert_angle(output), convert_angle(label)).item()

        avg_l1_loss /= len(self.eval_data)
        avg_angle_loss /= len(self.eval_data)
        return avg_l1_loss, avg_angle_loss

    def learn(self, train_data, eval_data, epochs, lr):
        self.train_data, self.eval_data = train_data, eval_data
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        train_l1_losses, train_angle_losses, eval_l1_losses, eval_angle_losses = [], [], [], []
        for epoch in tqdm(range(epochs)):
            train_l1_loss, train_angle_loss = self._train()
            eval_l1_loss, eval_angle_loss = self._eval()

            train_l1_losses.append(train_l1_loss)
            train_angle_losses.append(train_angle_loss)
            eval_l1_losses.append(eval_l1_loss)
            eval_angle_losses.append(eval_angle_loss)

            print(f"Epoch {epoch} results:")
            print(f"Training losses: l1 = {train_l1_loss}, deg = {train_angle_loss}")
            print(f"Evaluation losses: l1 = {eval_l1_loss}, deg = {eval_angle_loss}")

        print("Training losses:")
        print(f"L1: {train_l1_losses}")
        print(f"Angle: {train_angle_losses}")
        print("Evaluation losses:")
        print(f"L1: {eval_l1_losses}")
        print(f"Angle: {eval_angle_losses}")
