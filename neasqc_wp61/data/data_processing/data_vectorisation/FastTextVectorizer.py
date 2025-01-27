import sys
import json
import datetime
import argparse
import os

import tornado.ioloop
import tornado.web
import fasttext

class VectorizeRequestHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model

    def get(self):
        q = self.get_query_argument("q", "", False)
        words = q.split()
        response = [{'word': w,
                     'vector': self.model.get_word_vector(w).tolist()}
                   for w in words]
        self.write(json.dumps(response, ensure_ascii=False))

class ParamsRequestHandler(tornado.web.RequestHandler):
    def initialize(self, model, model_file_name):
        self.model = model
        self.model_file_name = model_file_name

    def get(self):
        response = {
            'type': 'fastText',
            'dim': self.model.get_dimension(),
            'model_file_name': self.model_file_name,
        }
        self.write(json.dumps(response, indent=2))

def make_app(model, model_file_name):
    return tornado.web.Application([
        (r"/vectorize", VectorizeRequestHandler, {"model": model}),
        (r"/params", ParamsRequestHandler, {"model": model, "model_file_name": model_file_name}),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="fasttext model file")
    parser.add_argument("--port", required=False, help="port to listen", type=int, default=8888)

    args = parser.parse_args()

    print("Starting:")
    print(f"model: {args.model}")
    print(f"port: {args.port}")

    model_file_name = os.path.basename(args.model)
    model = fasttext.load_model(args.model)
    
    app = make_app(model, model_file_name)
    app.listen(args.port)
    print("Server started!")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    sys.exit(int(main() or 0))






