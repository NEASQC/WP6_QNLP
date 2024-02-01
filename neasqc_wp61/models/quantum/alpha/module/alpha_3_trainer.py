import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from alpha_3_model import Alpha3Model
from utils import seed_everything, preprocess_train_test_dataset_for_alpha_3, BertEmbeddingDataset

class Alpha3Trainer():
    def __init__(self, number_of_epochs: int, train_path: str, val_path: str, test_path: str, seed: int, n_qubits: int, q_delta: float,
                 batch_size: int, lr: float, weight_decay: float, step_lr: int, gamma: float):
        
        self.number_of_epochs = number_of_epochs
        self.train_path = train_path
        self.val_path = val_path
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


        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = preprocess_train_test_dataset_for_alpha_3(self.train_path, self.val_path, self.test_path)


        self.n_classes = self.Y_train.apply(tuple).nunique()
        self.softmax = nn.Softmax()

        print("In the dataset there is:", self.n_classes, "classes")

        # initialise datasets and optimizers as in PyTorch

        self.train_dataset = BertEmbeddingDataset(self.X_train, self.Y_train)
        self.train_labels = torch.argmax(torch.tensor(self.train_dataset.Y.tolist()), dim=1).tolist()
        self.training_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)

        # Shuffle is set to False for the validation dataset because in the predict function we need to keep the order of the predictions
        self.validation_dataset = BertEmbeddingDataset(self.X_val, self.Y_val)
        self.validation_labels = torch.argmax(torch.tensor(self.validation_dataset.Y.tolist()), dim=1).tolist()
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size)

        self.test_dataset = BertEmbeddingDataset(self.X_test, self.Y_test)
        self.test_labels = torch.argmax(torch.tensor(self.test_dataset.Y.tolist()), dim=1).tolist()
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)


        # initialise the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # initialise model
        self.model = Alpha3Model(self.n_qubits, self.q_delta, self.n_classes, self.device)


        # initialise loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_lr, gamma=self.gamma)
	
        #self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.criterion.to(self.device)


        

    def train(self):
        train_loss_list = []
        train_acc_list = []
        train_preds_list = []
        train_probs_list = []

        val_loss_list = []
        val_acc_list = []
        val_preds_list = []
        val_probs_list = [] 

        best_val_acc = 0.0

        for epoch in range(self.number_of_epochs):
            print('Epoch: {}'.format(epoch))
            running_loss = 0.0
            running_corrects = 0

            self.model.train()
            #with torch.enable_grad():
            #for circuits, embeddings, labels in train_dataloader:
            train_preds_epoch = []
            train_probs_epoch = []
            for inputs, labels in self.training_dataloader:
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                
                _, preds = torch.max(self.softmax(outputs), 1)
                train_preds_epoch.append(preds)
                train_probs_epoch.append(self.softmax(outputs))
                loss = self.criterion(outputs, labels)
                loss.backward()

                # print('preds: ', preds)
                # print('labels: ', torch.max(labels, 1)[1])

                
            
                self.optimizer.step()

                # Print iteration results
                running_loss += loss.item()*batch_size_

                batch_corrects = torch.sum(preds == torch.max(labels, 1)[1]).item()
                running_corrects += batch_corrects

            
            # Print epoch results
            train_loss = running_loss / len(self.training_dataloader.dataset)
            train_acc = running_corrects / len(self.training_dataloader.dataset)
            train_preds_list.append(torch.cat(train_preds_epoch, dim=0).tolist())
            train_probs_list.append(torch.cat(train_probs_epoch, dim=0).tolist())
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            

            running_loss = 0.0
            running_corrects = 0

            self.model.eval()

            with torch.no_grad():
                val_preds_epoch = []
                val_probs_epoch = []
                for inputs, labels in self.validation_dataloader:
                    batch_size_ = len(inputs)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    _, preds = torch.max(self.softmax(outputs), 1)
                    val_preds_epoch.append(preds)
                    val_probs_epoch.append(self.softmax(outputs))
                    loss = self.criterion(outputs, labels)
                    
                    # Print iteration results
                    running_loss += loss.item()*batch_size_
                    batch_corrects = torch.sum(preds == torch.max(labels, 1)[1]).item()
                    running_corrects += batch_corrects
                
                val_preds_list.append(torch.cat(val_preds_epoch, dim=0).tolist())
                val_probs_list.append(torch.cat(val_probs_epoch, dim=0).tolist())


            validation_loss = running_loss / len(self.validation_dataloader.dataset)
            validation_acc = running_corrects / len(self.validation_dataloader.dataset)


            val_loss_list.append(validation_loss)
            val_acc_list.append(validation_acc)

            if validation_acc > best_val_acc:
                best_val_acc = validation_acc
                best_model = self.model.state_dict()
                

            self.lr_scheduler.step()
            
            print('Train loss: {}'.format(train_loss))
            print('Valid loss: {}'.format(validation_loss))
            print('Train acc: {}'.format(train_acc))
            print('Valid acc: {}'.format(validation_acc))

            print('-'*20)

        return (
            self.train_labels, self.validation_labels, self.test_labels,
            train_preds_list, val_preds_list, train_loss_list, train_acc_list,
            val_loss_list, val_acc_list, best_val_acc, best_model, train_probs_list,
            val_probs_list)


    def compute_test_logs(self, best_model):
        running_loss = 0.0
        running_corrects = 0

        # Load the best model found during training
        self.model.load_state_dict(best_model)
        self.model.eval()
        test_preds_epoch = []
        test_probs_epoch = []
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(self.softmax(outputs), 1)
                test_preds_epoch.append(preds)
                test_probs_epoch.append(self.softmax(outputs))
                loss = self.criterion(outputs, labels)
                
                # Print iteration results
                running_loss += loss.item()*batch_size_
                batch_corrects = torch.sum(preds == torch.max(labels, 1)[1]).item()
                running_corrects += batch_corrects

            test_preds = torch.cat(test_preds_epoch, dim=0).tolist()
            test_probs = torch.cat(test_probs_epoch, dim=0).tolist()
        test_loss = running_loss / len(self.test_dataloader.dataset)
        test_acc = running_corrects / len(self.test_dataloader.dataset)

        print('Run test results:')
        print('Test loss: {}'.format(test_loss))
        print('Test acc: {}'.format(test_acc))

        return test_preds, test_loss, test_acc, test_probs