FROM python:3.8-buster

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

ENV port="12345"
ENV model_file="/app/data/wiki.en.bin"

ENTRYPOINT ["sh", "-c", "python src/FastTextVectorizer.py ${model_file} --port ${port}"]
