FROM tensorflow/tensorflow:1.15.5-py3
RUN pip install bert-serving-server[http]

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

ENV port="12345"
ENV model_file="/app/data/BERTModels/cased_L-12_H-768_A-12"

ENTRYPOINT ["sh", "-c", "python src/BertVectorizer.py ${model_file} --port ${port}"]
