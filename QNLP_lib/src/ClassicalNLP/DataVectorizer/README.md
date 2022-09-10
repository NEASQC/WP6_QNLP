# Vectorizer

Services used for vectorizing using pretrained word embeddings.

The aim is to have vectorizing service detached from the rest of the library so that different vectorizing methods can easily be tested using the same interface.

Currently, vectorizing with `BERT` and `fastText` models are implemented.

### Setup

The vectorizing services are built as a `Docker` container and instructions for building the images are contained in the respective `Dockerfile`s. Therefore, a simple `docker build -t <service_name> .` command should suffice to build the image.
