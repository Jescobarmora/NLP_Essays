import numpy as np
from transformers import AutoTokenizer, AutoModel

def get_bert_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for text in texts:
        text_str = ' '.join(text)
        inputs = tokenizer(text_str, return_tensors='pt', truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)