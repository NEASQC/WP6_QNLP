import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

#NOTE: input file on which to FIT goes here
df_train = pd.read_csv("../datasets/agnews_balanced_train_sentence_bert.csv", sep='\t')
#NOTE: input file on which to apply the TRANSFORM goes here
df_test = pd.read_csv("../datasets/agnews_balanced_test_sentence_bert.csv", sep='\t')

# Convert the string representation to actual lists of embeddings
df_train["sentence_embedding"] = df_train["sentence_embedding"].apply(eval)
embeddings_train = df_train["sentence_embedding"].tolist()
embeddings_array_train = np.array(embeddings_train)

df_test["sentence_embedding"] = df_test["sentence_embedding"].apply(eval)
embeddings_to_transform = df_train["sentence_embedding"].tolist()
embeddings_array_to_transform = np.array(embeddings_to_transform)

pca = PCA(n_components=8)
pca.fit(embeddings_array_train) # fit on train
reduced_embeddings_test = pca.transform(embeddings_array_train) #transform on test using train fit
reduced_embeddings_train = pca.transform(embeddings_array_train) #transform on train using train fit

df_test["reduced_embedding"] = reduced_embeddings_test.tolist()
df_train["reduced_embedding"] = reduced_embeddings_train.tolist()

#NOTE: desired output file directory/location goes here
df_test.to_csv("../datasets/agnews_balanced_test_bert_pca.csv", sep='\t', index=False)
df_train.to_csv("../datasets/agnews_balanced_train_bert_pca.csv", sep='\t', index=False)


