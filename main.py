import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from datetime import datetime
from tqdm import tqdm
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from dataset import BERTDataset
from model import BERTClassifier
from utils import calc_accuracy, time_check

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SEED = 42

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data = pd.read_csv(os.path.join(DATA_DIR, 'train_g2p_removed_label_error_revised.csv'))
dataset_train, dataset_eval = train_test_split(data, train_size=0.7, random_state=SEED)

data_train = BERTDataset(dataset_train, tok, max_len, True, False)
data_eval = BERTDataset(dataset_eval, tok, max_len, True, False)

train_dataloader = DataLoader(data_train, batch_size=batch_size,num_workers=4)
eval_dataloader = DataLoader(data_eval, batch_size=batch_size,num_workers=4)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(DEVICE)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
now = datetime.now()
if __name__ == "__main__":
    time = time_check()
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader),
                                                                            total=len(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(DEVICE)
            segment_ids = segment_ids.long().to(DEVICE)
            valid_length = valid_length
            label = label.long().to(DEVICE)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                         train_acc / (batch_id + 1)))
        print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(eval_dataloader),
                                                                            total=len(eval_dataloader)):
            token_ids = token_ids.long().to(DEVICE)
            segment_ids = segment_ids.long().to(DEVICE)
            valid_length = valid_length
            label = label.long().to(DEVICE)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
        torch.save(model.state_dict(),'./save_folder/'+'model_state_dict.pth')
        print('./save_folder/'+'model_state_dict.pth')