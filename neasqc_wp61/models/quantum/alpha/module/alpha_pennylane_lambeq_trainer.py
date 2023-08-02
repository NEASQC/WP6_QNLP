from lambeq import BobcatParser, Rewriter, AtomicType, IQPAnsatz, remove_cups, Dataset
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from lambeq import CircuitAnsatz
from alpha_pennylane_lambeq_model import Alpha_pennylane_lambeq_model
from utils import seed_everything, preprocess_train_test_dataset, preprocess_train_test_dataset_words

class Alpha_pennylane_lambeq_trainer():
    def __init__(self, number_of_epochs: int, train_path: str, test_path: str, seed: int, 
                 ansatz: CircuitAnsatz, qn: int, qs: int, n_layers: int, n_single_qubit_params: int, 
                 batch_size: int, lr: float, weight_decay: float, step_lr: int, gamma: float, 
                 version_original: bool, reduced_word_embedding_dimension: int):
        
        self.number_of_epochs = number_of_epochs
        self.train_path = train_path
        self.test_path = test_path
        self.seed = seed
        self.ansatz = ansatz
        self.qn = qn
        self.n_layers = n_layers
        self.n_single_qubit_params = n_single_qubit_params
        self.qs = qs
        self.batch_size = batch_size

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_lr = step_lr
        self.gamma = gamma

        self.version_original = version_original
        self.reduced_word_embedding_dimension = reduced_word_embedding_dimension

        # seed everything
        seed_everything(self.seed)


        # load the dataset
        if version_original:
            self.X_train, self.X_test, self.Y_train, self.Y_test = preprocess_train_test_dataset_words(self.train_path, self.test_path, self.reduced_word_embedding_dimension)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = preprocess_train_test_dataset(self.train_path, self.test_path)

        self.train_diagrams, self.train_labels = self.create_diagrams(self.X_train, self.Y_train)
        self.test_diagrams, self.test_labels = self.create_diagrams(self.X_test, self.Y_test)


        train_circuits = self.create_circuits(self.train_diagrams, self.ansatz, self.n_layers, self.n_single_qubit_params, self.qn, self.qs)
        val_circuits = self.create_circuits(self.test_diagrams, self.ansatz, self.n_layers, self.n_single_qubit_params, self.qn, self.qs)

        self.all_circuits = train_circuits + val_circuits


        # initialise datasets and optimizers as in PyTorch
        if version_original:
            self.train_dataset = Dataset(list(zip(train_circuits, self.X_train['sentence_vectorized'].values.tolist())),
                                    self.Y_train,
                                    batch_size=self.batch_size)

            self.valid_dataset = Dataset(list(zip(val_circuits, self.X_test['sentence_vectorized'].values.tolist())),
                                    self.Y_test,
                                    batch_size=self.batch_size)
        else:
            self.train_dataset = Dataset(list(zip(train_circuits, np.vstack(self.X_train['sentence_embedding'].apply(np.array)))),
                                    self.Y_train,
                                    batch_size=self.batch_size)

            self.valid_dataset = Dataset(list(zip(val_circuits, np.vstack(self.X_test['sentence_embedding'].apply(np.array)))),
                                    self.Y_test,
                                    batch_size=self.batch_size)
        

        # initialise model
        self.model = Alpha_pennylane_lambeq_model.from_diagrams(self.all_circuits, probabilities=True, normalize=True, 
                                                                version_original = version_original, reduced_word_embedding_dimension = reduced_word_embedding_dimension)
        # initialise our model by pas sing in the diagrams, so that we have trainable parameters for each token
        self.model.initialise_weights()
        self.model.update_pre_qc_output_size()
        self.model.find_param_indexes_from_diagram_dict(self.all_circuits)
        self.model = self.model.double()

        # initialise loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_lr, gamma=self.gamma)
        
        # initialise the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion.to(self.device)

        

    def train(self):
        training_loss_list = []
        training_acc_list = []

        validation_loss_list = []
        validation_acc_list = []

        best_val_acc = 0

        for epoch in range(self.number_of_epochs):
            print('Epoch: {}'.format(epoch))
            running_loss = 0.0
            running_corrects = 0

            self.model.train()
            #for circuits, embeddings, labels in train_dataloader:
            for input, labels in self.train_dataset:
                batch_size_ = len(input)
                circuits, embeddings = np.array(input).T
                self.optimizer.zero_grad()

                if self.version_original:
                    # Word version
                    embedding_list = [torch.tensor(embedding) for embedding in embeddings]
                    predicted = self.model(circuits, embedding_list)
                else:
                    # Sentence version
                    predicted = self.model(circuits, torch.from_numpy(np.vstack(embeddings)))

                # use BCELoss as our outputs are probabilities, and labels are binary
                loss = self.criterion(torch.flatten(predicted), torch.DoubleTensor(labels))
                #running_loss += loss.item()
                running_loss += loss.item()*batch_size_
                loss.backward()
                self.optimizer.step()

                batch_corrects = (torch.round(torch.flatten(predicted)) == torch.DoubleTensor(labels)).sum().item()
                running_corrects += batch_corrects
                


            # Print epoch results
            train_loss = running_loss / len(self.train_dataset)
            train_acc = running_corrects / len(self.train_dataset)
            
            training_loss_list.append(train_loss)
            training_acc_list.append(train_acc)

            running_loss = 0.0
            running_corrects = 0

            self.model.eval()

            with torch.no_grad():
                for input, labels in self.valid_dataset:
                    batch_size_ = len(input)
                    circuits, embeddings = np.array(input).T
                    self.optimizer.zero_grad()
                    
                    if self.version_original:
                        # Word version
                        embedding_list = [torch.tensor(embedding) for embedding in embeddings]
                        predicted = self.model(circuits, embedding_list)
                    else:
                        # Sentence version
                        predicted = self.model(circuits, torch.from_numpy(np.vstack(embeddings)))

                    loss = self.criterion(torch.flatten(predicted), torch.DoubleTensor(labels))
                    running_loss += loss.item()*batch_size_


                    batch_corrects = (torch.round(torch.flatten(predicted)) == torch.DoubleTensor(labels)).sum().item()
                    running_corrects += batch_corrects


            validation_loss = running_loss / len(self.valid_dataset)
            validation_acc = running_corrects / len(self.valid_dataset)


            validation_loss_list.append(validation_loss)
            validation_acc_list.append(validation_acc)

            if validation_acc > best_val_acc:
                best_val_acc = validation_acc
                best_model = self.model.state_dict()


            self.lr_scheduler.step()
            
            print('Train loss: {}'.format(train_loss))
            print('Valid loss: {}'.format(validation_loss))
            print('Train acc: {}'.format(train_acc))
            print('Valid acc: {}'.format(validation_acc))

        return training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, best_val_acc, best_model


    def predict(self):
        prediction_list = torch.tensor([])
        
        self.model.eval()

        with torch.no_grad():
            for input, labels in self.valid_dataset:
                circuits, embeddings = np.array(input).T

                if self.version_original:
                    # Word version
                    embedding_list = [torch.tensor(embedding) for embedding in embeddings]
                    predicted = self.model(circuits, embedding_list)
                else:
                    # Sentence version
                    predicted = self.model(circuits, torch.from_numpy(np.vstack(embeddings)))

                prediction_list = torch.cat((prediction_list, torch.round(torch.flatten(predicted))))

        return prediction_list.detach().cpu().numpy()


    def create_diagrams(self, X: list, Y: list):
        parser = BobcatParser(root_cats=('NP', 'N'), verbose='text')

        raw_diagrams = parser.sentences2diagrams(X['sentence'].values.tolist(), suppress_exceptions=True)

        # Apply rewrite rule for prepositional phrases
        # Here we use try and except to avoid errors when the parser fails to parse a sentence (occurs only for the 21k dataset)
        rewriter = Rewriter(['prepositional_phrase', 'determiner', 'curry'])

        raw_diagrams_rewrited = []
        y_rewrited = []

        for index, diagram in enumerate(raw_diagrams):
            try:
                raw_diagrams_rewrited.append(rewriter(diagram))
                y_rewrited.append(Y.iloc[index])
            except TypeError:
                print(index)
                print("Error in rewriting diagram: ", diagram)


        raw_diagrams = raw_diagrams_rewrited
        Y = y_rewrited

        diagrams = [
            diagram.normal_form()
            for diagram in raw_diagrams if diagram is not None
        ]

        labels = [
            label for (diagram, label)
            in zip(diagrams, Y)
            if diagram is not None]

        return diagrams, labels

    
    def create_circuits(self, diagrams: list, ansatz: CircuitAnsatz, n_layers: int, n_single_qubit_params: int, qn: int, qs: list):
        """
        Create the circuits from the diagrams
        """
        ansatz = ansatz({AtomicType.NOUN: qn, AtomicType.SENTENCE: qs},
                        n_layers=n_layers, n_single_qubit_params=n_single_qubit_params)
        
        circuits = [ansatz(remove_cups(diagram)) for diagram in diagrams]

        return circuits






