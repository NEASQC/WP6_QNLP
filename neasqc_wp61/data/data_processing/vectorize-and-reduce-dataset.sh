dataset=../datasets/ag_news_balanced
python dataset_vectoriser.py ${dataset}_test.tsv -e sentence
python dataset_vectoriser.py ${dataset}_train.tsv -e sentence
python dataset_vectoriser.py ${dataset}_dev.tsv -e sentence

python generate_pca_test_dataset.py "${dataset}_train_sentence_bert.csv" "${dataset}_test_sentence_bert.csv" "${dataset}_train_sentence_bert_pca.csv" "${dataset}_test_sentence_bert_pca.csv"

python generate_fasttext_dataset.py "${dataset}_test_sentence_bert.csv" "${dataset}_test_sentence_bert_ft.csv"
python generate_fasttext_dataset.py "${dataset}_train_sentence_bert.csv" "${dataset}_train_sentence_bert_ft.csv"
python generate_fasttext_dataset.py "${dataset}_dev_sentence_bert.csv" "${dataset}_dev_sentence_bert_ft.csv"

