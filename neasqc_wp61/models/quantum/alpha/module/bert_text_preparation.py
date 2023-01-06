from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_text_preparation(text: str, tokenizer = tokenizer)->tuple:
    """Tokenises sentence.

    Uses Bert Tokeniser to tokenise a sentence(string). It returns the tokenized text, tokens tensor and segments tensors.:

    Parameters
    ----------
    text : str
        A sentence to be tokenised.
    tokeniser : tokenizer
        The tokeniser being used. The default is set to the transformers pretrained bert tokeniser.

    Returns
    -------
    (tokenized_text, tokens_tensor, segments_tensors): tuple  

    """
    
    
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    del_list = []
    for i,x in enumerate(tokenized_text):
        if x[0]=='#':
            tokenized_text[i] = tokenized_text[i-1] + tokenized_text[i][2:]
            del_list.append(i-1)
    tokenized_text = [tokenized_text[i] for i in range(len(tokenized_text)) if i not in del_list]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)


    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors