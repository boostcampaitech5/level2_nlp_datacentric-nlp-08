import pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.dataset import health_summary
import numpy as np
import ast

ori_dataset = pd.read_csv('../data/train_g2p_removed_label_error_revised.csv')
dataset = pd.read_csv('../output/train_output.csv')
pred_probs = dataset['class_preds']

new_pred_probs = np.array(eval(pred_probs[0])).reshape(-1,7)
for i in range(1,len(pred_probs)):
    temp = np.array(eval(pred_probs[i])).reshape(-1,7)
    new_pred_probs = np.concatenate((new_pred_probs,temp),axis=0)

label = np.array(ori_dataset['target'])

ordered_label_issues = find_label_issues(
    labels= label,
    pred_probs=new_pred_probs,
    return_indices_ranked_by= 'self_confidence'
)
head_issues = ordered_label_issues

print(head_issues)
print(len(head_issues))
for issue in head_issues:
    print('input_Text :',ori_dataset.iloc[issue]['input_text'],issue)
    print('label:',ori_dataset.iloc[issue]['label_text'])
class_names=[0,1,2,3,4,5,6]
health_summary(label, new_pred_probs, class_names=class_names)