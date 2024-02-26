import unittest 
import os 
import sys 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/alpha/module/")
from alpha_3_trainer import Alpha3Trainer
from alpha_3_model import Alpha3Model
from torch.utils.data import DataLoader
from utils import seed_everything, preprocess_train_test_dataset_for_alpha_3, BertEmbeddingDataset

class TestAlpha3(unittest.TestCase):
    """
    Class for testing Alpha3 model
    """
    
    def setUp(self):
        self.iterations = 2
        self.train_path = ("./../neasqc_wp61/data/" +
        "toy_dataset/toy_dataset_bert_sentence_embedding_train.csv")
        self.val_path = ("./../neasqc_wp61/data/" +
        "toy_dataset/toy_dataset_bert_sentence_embedding_val.csv")
        self.test_path = ("./../neasqc_wp61/data/" +
        "toy_dataset/toy_dataset_bert_sentence_embedding_test.csv")
        self.seed = 18051967
        self.n_qubits = 3
        self.q_delta = 0.01
        self.batch_size = 32
        self.lr = 2e-03
        self.weight_decay = 0.0
        self.step_lr = 20
        self.gamma = 0.5
        self.trainer = Alpha3Trainer(
            self.iterations, self.train_path, self.val_path, self.test_path,
            self.seed, self.n_qubits, self.q_delta, self.batch_size, self.lr,
            self.weight_decay, self.step_lr, self.gamma
        )
        self.model = Alpha3Model(
            self.n_qubits, self.q_delta, self.trainer.n_classes,
            self.trainer.device)

        ### Create torch datasets
        X_Y_values = preprocess_train_test_dataset_for_alpha_3(
            self.train_path, self.val_path, self.test_path)
        self.X_train = X_Y_values[0]
        self.Y_train = X_Y_values[3]
        self.train_dataset = BertEmbeddingDataset(self.X_train, self.Y_train)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size = self.batch_size
        )
        self.input_sample = list(self.train_dataloader)[0][0]


    def test_device(self):
        self.assertIn(self.trainer.device.type, ['cpu', 'cuda:0'])

    def test_n_classes(self):
        self.assertIs(type(self.trainer.n_classes), int)

    def test_model(self):
        self.assertEqual(self.model.pre_net.out_features, self.n_qubits)
        self.assertEqual(
            self.model.post_net.out_features, self.trainer.n_classes)
        model_results = self.model.forward(self.input_sample)
        print(dir(model_results.shape))
        self.assertEqual(model_results.shape, (self.batch_size, 2))

    def test_train_output(self):        
        trainer_results = self.trainer.train()
        train_loss = trainer_results[0]
        train_acc = trainer_results[1]
        val_loss = trainer_results[2]
        val_acc = trainer_results[3]
        self.assertEqual(len(train_loss), self.iterations)
        self.assertEqual(len(train_acc), self.iterations)
        self.assertEqual(len(val_loss), self.iterations)
        self.assertEqual(len(val_acc), self.iterations)
        for i in range(self.iterations):
            self.assertLessEqual(train_acc[i], 1)
            self.assertLessEqual(val_acc[i], 1)




if __name__ == "__main__":
    unittest.main()