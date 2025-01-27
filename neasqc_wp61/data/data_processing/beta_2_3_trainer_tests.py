import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import joblib

# from torchinfo import summary
from beta_2_3_model import Beta_2_3_model
from utils import (
    seed_everything,
    EmbeddingDataset,
    encode_labels,
    load_json_data,
    extract_features_labels,
    create_pandas_series,
    create_dataloader
)


class Beta_2_3_trainer_tests:
    def __init__(
        self,
        optimiser: str,
        run_number: int,
        number_of_epochs: int,
        train_path: str,
        valid_path: str,
        test_path: str,
        seed: int,
        n_qubits: int,
        q_delta: float,
        batch_size: int,
        lr: float,
        weight_decay: float,
        step_lr: int,
        gamma: float,
    ):

        self.optimiser = optimiser
        self.run_number = run_number
        self.number_of_epochs = number_of_epochs
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.seed = seed
        self.n_qubits = n_qubits
        self.q_delta = q_delta
        self.batch_size = batch_size

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_lr = step_lr
        self.gamma = gamma

        # seed everything
        seed_everything(self.seed)
        
        train_raw_data = load_json_data(self.train_path)
        valid_raw_data = load_json_data(self.valid_path)
        test_raw_data = load_json_data(self.test_path)
        
        X_train_raw, Y_train_raw = extract_features_labels(train_raw_data)
        X_valid_raw, Y_valid_raw = extract_features_labels(valid_raw_data)
        X_test_raw, Y_test_raw = extract_features_labels(test_raw_data)
        
        Y_train_encoded, self.label_encoder = encode_labels(Y_train_raw)
        Y_valid_encoded = self.label_encoder.transform(Y_valid_raw)
        Y_test_encoded = self.label_encoder.transform(Y_test_raw)
        
        self.X_train_series, self.Y_train_series = create_pandas_series(X_train_raw, Y_train_encoded)
        self.X_valid_series, self.Y_valid_series = create_pandas_series(X_valid_raw, Y_valid_encoded)
        self.X_test_series, self.Y_test_series = create_pandas_series(X_test_raw, Y_test_encoded)

        self.n_classes = self.Y_train_series.nunique()

        print("There are:", self.n_classes, "classes in the dataset.")

        # initialise datasets and optimizers as in PyTorch

        self.train_dataset = EmbeddingDataset(self.X_train_series, self.Y_train_series)
        self.training_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Shuffle is set to False for the validation dataset because in the predict function we need to keep the order of the predictions
        self.valid_dataset = EmbeddingDataset(self.X_valid_series, self.Y_valid_series)
        self.valid_dataloader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Shuffle is set to False for the test dataset because in the predict function we need to keep the order of the predictions
        self.test_dataset = EmbeddingDataset(self.X_test_series, self.Y_test_series)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # initialise the device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # initialise model
        self.model = Beta_2_3_model(
            self.n_qubits, self.q_delta, self.n_classes, self.device
        )

        # summary(self.model)

        # initialise loss and optimizer
        self.criterion = nn.CrossEntropyLoss()

        if self.optimiser == "Adam":
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "RMSprop":
            self.opt = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "Adadelta":
            self.opt = torch.optim.Adadelta(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "Adagrad":
            self.opt = torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "AdamW":
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "Adamax":
            self.opt = torch.optim.Adamax(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "ASGD":
            self.opt = torch.optim.ASGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "LBFGS":
            self.opt = torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "NAdam":
            self.opt = torch.optim.NAdam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "RAdam":
            self.opt = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "Rprop":
            self.opt = torch.optim.Rprop(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimiser == "SGD":
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        self.lr_scheduler = lr_scheduler.StepLR(
            self.opt, step_size=self.step_lr, gamma=self.gamma
        )

        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def save_label_encoder(self, out_pkl_path):
        joblib.dump(self.label_encoder, out_pkl_path)

    def train(self):
        training_loss_list = []
        training_acc_list = []

        valid_loss_list = []
        valid_acc_list = []

        best_valid_acc = 0.0

        for epoch in range(self.number_of_epochs):
            print("Epoch: {}".format(epoch))
            running_loss = 0.0
            running_corrects = 0

            self.model.train()
            # with torch.enable_grad():
            # for circuits, embeddings, labels in train_dataloader:
            for inputs, labels in self.training_dataloader:
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # print('preds: ', preds)
                # print('labels: ', torch.max(labels, 1)[1])

                self.opt.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_

                batch_corrects = torch.sum(
                    preds == labels
                ).item()
                running_corrects += batch_corrects

            # Print epoch results
            train_loss = running_loss / len(self.training_dataloader.dataset)
            train_acc = running_corrects / len(
                self.training_dataloader.dataset
            )

            training_loss_list.append(train_loss)
            training_acc_list.append(train_acc)

            running_loss = 0.0
            running_corrects = 0

            self.model.eval()

            with torch.no_grad():
                for inputs, labels in self.valid_dataloader:
                    batch_size_ = len(inputs)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.opt.zero_grad()

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    # Print iteration results
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(
                        preds == labels
                    ).item()
                    running_corrects += batch_corrects

            valid_loss = running_loss / len(self.valid_dataloader.dataset)
            valid_acc = running_corrects / len(self.valid_dataloader.dataset)

            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model = self.model.state_dict()

            self.lr_scheduler.step()

            print("Train loss: {}".format(train_loss))
            print("Valid loss: {}".format(valid_loss))
            print("Train acc: {}".format(train_acc))
            print("Valid acc: {}".format(valid_acc))

            print("-" * 20)

        return (
            training_loss_list,
            training_acc_list,
            valid_loss_list,
            valid_acc_list,
            best_valid_acc,
            best_model
        )

    def predict(self):
        prediction_list = torch.tensor([]).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                prediction_list = torch.cat(
                    (prediction_list, torch.round(torch.flatten(preds)))
                )

        return prediction_list.detach().cpu().numpy()

    def compute_test_logs(self, best_model):
        running_loss = 0.0
        running_corrects = 0

        # Load the best model found during training
        self.model.load_state_dict(best_model)
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.opt.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Print iteration results
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(
                    preds == labels
                ).item()
                running_corrects += batch_corrects

        test_loss = running_loss / len(self.test_dataloader.dataset)
        test_acc = running_corrects / len(self.test_dataloader.dataset)

        print("Run test results:")
        print("Test loss: {}".format(test_loss))
        print("Test acc: {}".format(test_acc))

        return test_loss, test_acc
