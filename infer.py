import os

import pandas as pd
import torch
from kobert import get_pytorch_kobert_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BERTDataset
from main import DATA_DIR, tok, max_len, batch_size, DEVICE, BASE_DIR
from model import BERTClassifier

dataset_eval = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
dataset_eval['target'] = [0]*len(dataset_eval)
data_eval = BERTDataset(dataset_eval, tok, max_len, True, False)
eval_dataloader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    preds = []
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(DEVICE)
    model.load_state_dict(torch.load('./save_folder/2023_05_24_09_57_25_4_epoch_model_state_dict.pth'))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, _) in tqdm(enumerate(eval_dataloader),
                                                                    total=len(eval_dataloader)):
        token_ids = token_ids.long().to(DEVICE)
        segment_ids = segment_ids.long().to(DEVICE)
        valid_length = valid_length
        out = model(token_ids, valid_length, segment_ids)
        max_vals, max_indices = torch.max(out, 1)
        preds.extend(list(max_indices))

    preds = [int(p) for p in preds]

    dataset_eval['target'] = preds
    dataset_eval.to_csv(os.path.join('output', 'output.csv'), index=False)