An implementation of DisCoCat model for asserting the thruth value of sentences.

* sentence.py: **Sentence** class contains all the information for a sentence to be written as a quantum circuit: the number of qubits per word, the categories assigned to each part of speech, the parameters for the rotations and the qubits that need to be contracted. Also, some methods are provided to manipulate the parameters of sentences.

* dictionary.py:  Provides the information and parameters of the DisCoCat model. The different parameterizations, Ansatz shapes, and grammar-related information is contained in **QuantumDict** and **QuantumWord** classes.

* circuit.py: **CirctuiBuilder** class can use sentences to build the quantum circuit as QLM programs, submit jobs and apply the needed post-processing to sample output states. 

* optimizer.py: **ClassicalOptimizer** contains the cost functions for quantum circuits, and ue SciPy optimizers to train quantum circuits.

* loader.py: Used to read the .json files and create DataFrames used in the training process.

Currently, supported sentence structures are:

    - Noun+TransitiveVerb+Noun 
    - Adjective+Noun+TransitiveVerb+Adjective+Noun 
    - Noun+IntransitiveVerb+Preposition+Noun 
    - Noun+IntransitiveVerb

Contractions for these structures are done following the information contained in dictionaries within the code when the sentence type is given. As sentences grow in size and diversity, pregroup rules must be programmed to perform the needed contractions automatically.
