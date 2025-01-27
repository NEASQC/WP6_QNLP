import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Load the datasets
train_path = "./data/datasets/agnews_balanced_bert.csv"
test_path = "./data/datasets/agnews_balanced_test_bert.csv"
train_df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')

# Ensure the sentence embeddings are in the correct format (list of floats)
train_df["sentence_embedding"] = train_df["sentence_embedding"].apply(eval)
test_df["sentence_embedding"] = test_df["sentence_embedding"].apply(eval)

# Convert the sentence embeddings to numpy arrays for PCA
X_train = np.array(train_df["sentence_embedding"].tolist())
X_test = np.array(test_df["sentence_embedding"].tolist())

# Fit PCA on the training data
pca = PCA(n_components=8)
X_train_reduced = pca.fit_transform(X_train)

# Transform the test data using the fitted PCA
X_test_reduced = pca.transform(X_test)

# Add the reduced embeddings to the dataframes
train_df["reduced_embedding"] = X_train_reduced.tolist()
test_df["reduced_embedding"] = X_test_reduced.tolist()

# Save the updated datasets
train_df.to_csv(
    "./data/datasets/agnews_balanced_bert_train_pca_tests.csv", sep='\t', index=False
)
test_df.to_csv(
    "./data/datasets/agnews_balanced_bert_test_pca_tests.csv", sep='\t', index=False
)

print("Transformation complete and files saved.")
