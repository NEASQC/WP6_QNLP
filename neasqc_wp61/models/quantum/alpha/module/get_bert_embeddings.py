from transformers import BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )

def get_bert_embeddings(tokens_tensor, segments_tensors, model = model)->list:
    """Word embeddings for each word in a sentence.

    Returns word embeddings for each token in a sentence.:

    Parameters
    ----------
    tokens_tensor:tokens_tensor
        Tensor of tokens for a sentence
        
    segments_tensors:segments_tensor
        Tensor of segments of a sentence
        
    model:Embedding_Model
        Word embedding model to be used. The default is set to the Bert Model.
    
    

    Returns
    -------
    list_token_embeddings: list
        List consisting of word embeddings for each token in the sentence.
        
    """

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        hidden_states = outputs[2][1:]


    token_embeddings = hidden_states[-1]

    token_embeddings = torch.squeeze(token_embeddings, dim=0)

    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings