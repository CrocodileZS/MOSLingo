from transformers import BertModel
import torch
from dataloader.EvalDataLoader import TestDataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.eval_utils import run_eval
import time
from utils.log_utils import get_logger

# configurations
plm_for_tokenize = "bert-base-cased"

def main():
    test_data_loader = TestDataLoader(plm_for_tokenize=plm_for_tokenize,
                                 max_token_len=128)
    test_data_loader.add_ind_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/CLINIC/demo_clinic.csv")
    # test_data_loader.add_ind_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/M-CID/demo_mcid.csv")
    # test_data_loader.add_ind_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/HWU/demo_hwu.csv")
    # test_data_loader.add_ind_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/Snips/demo_snips.csv")
    test_data_loader.add_ood_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/CLINIC/demo_stackoverflow.csv")

    input_ids = torch.tensor([i for i in test_data_loader.test_input_ids_batch], dtype=torch.long)
    attention_mask = torch.tensor([i for i in test_data_loader.test_attention_mask_batch], dtype=torch.long)
    label_ids = torch.tensor([i for i in test_data_loader.test_labels_batch], dtype=torch.long)
    test_dataset = TensorDataset(input_ids, attention_mask, label_ids)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    group_slice = torch.load("./group_slice.pt")

    model = BertModel.from_pretrained("bert-base-cased")
    model.to(device)

    checkpoint = torch.load('<finetuned_model>.pth.tar')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    logger = get_logger(f'./eval_{time.time()}.log')

    for batch_index, batch in enumerate(test_dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # forward
        x = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        # set a linear for classification
        linear_input_size = model.embeddings.position_embeddings.embedding_dim  # last layer dimension
        linear_output_size = group_slice[-1][1] + 1
        linear = nn.Linear(in_features=linear_input_size, out_features=linear_output_size)
        linear.load_state_dict(checkpoint['linear'])

        logits = linear(x)
        pred_prob = F.softmax(logits)

        # pred
        run_eval(logger, pred_prob, group_slice, labels)

