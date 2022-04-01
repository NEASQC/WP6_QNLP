# Neural Networks in Dressed Quantum Circuits for NLP


## Dressed Quantum Circuits

The idea behind this Jupyter Notebook is to extend the work found in https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html. In transfer learning, the first layers of a pretrained neural network are used to solve a different problem to that used to train the network, adding new layers to the model that specialise in that specific task. 

We are using a BERT model to retrieve the context dependant embeddings for the words present in a sentence. Which layer is the best to retrieve the embeddings from is unclear, and it will need to be investigated. Once we have those vectors, they opropagate through a feedforward network that first will reduce the dimensionaility to an intermediate representation (in the notebook it is set to 20), and then the following layers will continue reducing the dimensionality of the vectors until reaching the number of parameters needed by the tensor representation of the quantum circuit offered by the Lambeq library for that word in a specific sentences. 

Some benefits of this approach are:

* Any sentence structure and word contained in the BERT model used can be processed by the full pipeline. No need to store the values of parameters for a dictionary
    
* It is possible to generalize to different NLP tasks
    
* If the dimensionaility of the category space is changed, the NN can be re-scaled to reuse the model for new circuit dimensionaility.
    
    

Issues
    
* Pytket loses the tensor nature of parameters, giving an output consisting of a list of floats or simply counts -> Differentiable circuits in Pennylane could be a solution.
    
* It is not clear if we gain any quantum advantage with this methods, as a classical NN has to be trained. 




