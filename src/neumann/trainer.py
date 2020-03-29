import time
from pathlib import Path

import torch
import torch.nn as nn

from src.neumann.data_utils import CIFAR10_CLASSES
from src.neumann.utils import SAVE_LOAD_TYPE, load_model, save_model, isclose


class Trainer(nn.Module):
    def __init__(self, model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config):
        super(Trainer, self).__init__()
        self.model_name = model_name
        self.model = model.to(config["device"])
        self.config = config
        self.data_loader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = 0
        self.max_epochs = config["num_of_train_epochs"]
        self.run_id = str(run_id)
        self.add_run_id = add_run_id

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        max_loss_repeat = 4
        loss_repeat_counter = 1
        prev_loss = float("-inf")
        for e in range(self.max_epochs):

            epoch_loss = self.train(e)

            if epoch_loss < max_loss:
                print("Loss decreased!")
                file_path = Path(".") / ("models/" + self.model_name)
                if self.config["save_model"] == SAVE_LOAD_TYPE.MODEL:
                    save_model(file_path, self.model, self.optimizer, self.run_id, self.add_run_id)
                max_loss = epoch_loss

            print("[TRAINING]  Epoch [%d/%d]   Loss: %.4f" % (self.epochs, self.max_epochs, epoch_loss))

            if isclose(epoch_loss, prev_loss, rel_tol=1e-4):
                loss_repeat_counter += 1
                if loss_repeat_counter >= max_loss_repeat:
                    print("Loss not decreasing for last {} times".format(loss_repeat_counter))
                    break
                else:
                    loss_repeat_counter += 1
            prev_loss = epoch_loss

            self.epochs += 1

        print("Total Training in mins: %5.2f" % ((time.time() - start_time) / 60))

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        for batch_num, data in enumerate(self.data_loader, 0):
            loss = self.train_batch(data)
            epoch_loss += loss.item()
            if batch_num % 2000 == 1999:  # print every 2000 mini-batches
                print("epoch:{:d}, batch:{:d}, loss:{:8.4f}"
                      .format(epoch + 1, batch_num + 1, epoch_loss / 2000))
                epoch_loss = 0.0

        return epoch_loss

    def train_batch(self, data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def test(self):
        file_path = Path(".") / ("models/" + self.model_name)
        if self.config["reload_model"] == SAVE_LOAD_TYPE.MODEL:
            model, optimizer = load_model(file_path, self.optimizer)
            self.model = model
            self.optimizer = optimizer
        correct = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        total = 0
        with torch.no_grad():
            for data in self.data_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        print("Accuracy of the network on the 10000 test images: %d %%" % (
                100 * correct / total))

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                CIFAR10_CLASSES[i], 100 * class_correct[i] / class_total[i]))
