from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        texts = dataset['input_text'].tolist()
        targets = dataset['target'].tolist()

        self.sentences = [transform([i]) for i in texts]
        self.labels = [np.int32(i) for i in targets]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))
