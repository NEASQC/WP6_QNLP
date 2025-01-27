import requests


class VectorizerConnector(object):
    """A class used to connect to a vectorizer service"""

    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.urlPrefix = f"{self.address}:{self.port}/"
        pass



    def vectorize_sentence(self, sentence):
        payload = {"q": sentence}
        r = requests.get(self.urlPrefix + "vectorize", params=payload)
        return r.json()



