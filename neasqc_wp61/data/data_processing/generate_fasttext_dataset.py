import pandas as pd
import fasttext as ft
import fasttext.util as ftu

# Load the dataframe
df = pd.read_csv("../datasets/agnews_balanced_test.csv", sep='\t')

# Rename the "text" column to "sentence"
# df.rename(columns={"text": "sentence"}, inplace=True)

ftu.download_model("en", if_exists="ignore")
model = ft.load_model("cc.en.300.bin")
ftu.reduce_model(model, 8)

reduced_embedding_list = []

for sentence in df.sentence.values:
    embedding = model.get_sentence_vector(sentence)
    reduced_embedding_list.append(embedding)

df["reduced_embedding"] = reduced_embedding_list

embeddings = list(df["reduced_embedding"].apply(lambda x: str(x)))
out_embeddings = []

for embedding in embeddings:
    cleaned_str = embedding.strip()[1:-1].strip()
    out_embedding = list(map(float, cleaned_str.split()))
    out_embeddings.append(out_embedding)

df["reduced_embedding"] = out_embeddings

df.to_csv("../datasets/agnews_balanced_test_ft.csv", sep='\t', index=False)
