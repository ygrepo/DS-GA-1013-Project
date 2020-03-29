import time
from pathlib import Path

import torch
import torch.nn as nn

from src.neumann.utils import SAVE_LOAD_TYPE


class Trainer(nn.Module):
    def __init__(self, model_name, model, optimizer, criterion,
                 train_dataloader, test_dataloader, run_id, config):
        super(Trainer, self).__init__()
        self.model_name = model_name
        self.model = model.to(config["device"])
        self.config = config
        self.training_data_loader = train_dataloader
        self.test_data_loader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = 0
        self.max_epochs = config["num_of_train_epochs"]
        self.run_id = str(run_id)
        self.add_run_id = config["add_run_id"]

    def train_epochs(self):
        best_acc = 0  # best test accuracy
        start_epoch = 0

        file_path = Path(".") / ("models/" + self.model_name.value)
        if self.config["reload_model"] == SAVE_LOAD_TYPE.MODEL:
            model_path = file_path / "model.pyt"
            if model_path.exists():
                if torch.cuda.is_available():
                    map_location = lambda storage, loc: storage.cuda()
                else:
                    map_location = "cpu"
                checkpoint = torch.load(model_path, map_location=map_location)
                self.model = checkpoint["model"]
                best_acc = checkpoint["acc"]
                start_epoch = checkpoint["epoch"]
                print(f"Restored checkpoint(whole model and optimizer state dictionary) from {model_path}.")

        start_time = time.time()
        for epoch in range(start_epoch, start_epoch + self.max_epochs):

            self.train(epoch)
            acc = self.test(epoch)

            # Save checkpoint.
            if acc > best_acc:
                print('Saving..')
                file_path = Path(".") / ("models/" + self.model_name.value)
                file_path.mkdir(parents=True, exist_ok=True)
                state = {
                    "model": self.model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                model_path = file_path / "model.pyt"
                torch.save(state, model_path)
                best_acc = acc

        print("Total Training in mins: %5.2f" % ((time.time() - start_time) / 60))

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0
        for batch_num, data in enumerate(self.training_data_loader):
            loss, total, correct = self.train_batch(total, correct, data)
            train_loss += loss.item()

            if batch_num % 100 == 99:  # print every 2000 mini-batches
                print("[TRAINING] epoch:{:d}, batch:{:d}, loss:{:8.4f}, acc:{:.3f}, ({:d}/{:d})"
                      .format(epoch + 1, batch_num + 1, train_loss / (batch_num + 1), 100. * correct / total, correct,
                              total))
                train_loss = 0.0

            # progress_bar(batch_num, len(self.training_data_loader),
            #              "Training Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})"
            #              .format(train_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return train_loss

    def train_batch(self, total, correct, data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs, targets = inputs.to(self.config["device"]), targets.to(self.config["device"])

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        return loss, total, correct

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, (inputs, targets) in enumerate(self.test_data_loader):
                inputs, targets = inputs.to(self.config["device"]), targets.to(self.config["device"])
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if batch_num % 100 == 99:  # print every 2000 mini-batches
                    print("[TESTING] epoch:{:d}, batch:{:d}, loss:{:8.4f}, acc:{:.3f}, ({:d}/{:d})"
                          .format(epoch + 1, batch_num + 1, test_loss / (batch_num + 1), 100. * correct / total,
                                  correct, total))

                    test_loss = 0.0

        return 100. * correct / total
