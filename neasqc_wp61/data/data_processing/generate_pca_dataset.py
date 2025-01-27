import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

#NOTE : input files goes here
df = pd.read_csv("../datasets/agnews_balanced_train_sentence_bert.csv", sep='\t')

# Convert the string representation to actual lists of embeddings
df["sentence_embedding"] = df["sentence_embedding"].apply(eval)

# Map 10 folds to 5 folds - makes code simpler
mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
df["split"] = df["split"].map(mapping)

embeddings = df["sentence_embedding"].tolist()
split_ids = df["split"].tolist()
embeddings_array = np.array(embeddings)

for i in range(5):
    # For each split, define the train data
    train = df[df["split"] != i]
    pca = PCA(n_components=8)
    # Fit PCA on train data
    fit_data = np.vstack(train["sentence_embedding"])
    pca.fit(fit_data)
    # Transform all data with fitted PCA
    reduced_embeddings = pca.transform(embeddings_array)
    # Add this to a specific column for that particular fold
    df[f"reduced_embedding_{i}"] = reduced_embeddings.tolist()

#NOTE: desired output file directory/location goes here
df.to_csv("../datasets/agnews_balanced_bert_pca.csv", sep='\t', index=False)
