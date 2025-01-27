import sys
import json
import datetime
import argparse
import os
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


import tornado.ioloop
import tornado.web

class VectorizeRequestHandler(tornado.web.RequestHandler):
    def initialize(self, bc):
        self.bc = bc

    def get(self):
        q = self.get_query_argument("q", "", False)
        res = self.bc.encode([q])

        self.write(json.dumps(res.tolist(), ensure_ascii=False))

class ParamsRequestHandler(tornado.web.RequestHandler):
    def initialize(self, bc, model_file_name):
        self.bc = bc
        self.model_file_name = model_file_name

    def get(self):
        response = {
            'type': 'BERT',
            'model_file_name': self.model_file_name,
            'config': self.bc.server_config,
        }
        self.write(json.dumps(response, indent=2))


def create_server(model_path):
    args = get_args_parser().parse_args(['-model_dir', model_path,
                                         '-cpu'])
    print(args)
    server = BertServer(args)
    return server

def make_app(bc, model_file_name):
    return tornado.web.Application([
        (r"/vectorize", VectorizeRequestHandler, {"bc": bc}),
        (r"/params", ParamsRequestHandler, {"bc": bc, "model_file_name": model_file_name}),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="bert model folder")
    parser.add_argument("--port", required=False, help="port to listen", type=int, default=8888)

    args = parser.parse_args()

    
    print("Starting:")
    print(f"model: {args.model}")
    print(f"port: {args.port}")
    print(sys.version)
    import tensorflow as tf; print(tf.__version__)

    model_file_name = os.path.basename(args.model)
    
    server = create_server(args.model)
    print("s1")
    server.start()
    print("s2")
    

    bc = BertClient()
    print("s3")
    
    app = make_app(bc, model_file_name)
    app.listen(args.port)
    print("Server started!")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    sys.exit(int(main() or 0))






