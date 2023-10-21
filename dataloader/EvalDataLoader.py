from transformers import BertTokenizer
import numpy as np


class TestDataLoader:
    def __init__(self,
                 plm_for_tokenize,
                 max_token_len):
        self.plm = plm_for_tokenize
        self.tokenizer = BertTokenizer.from_pretrained(self.plm)
        self.test_input_ids_batch = []
        self.test_attention_mask_batch = []
        self.test_labels_batch = []
        self.max_token_len = max_token_len

    def add_ind_data(self, path):
        with open(path) as file:
            lines = file.readlines()
        for line in lines:
            try:
                group_id, class_id, sent = line.strip().split(',', 2)
            except ValueError:
                continue
            tokenized_data = self.tokenizer.encode_plus(sent,
                                                        add_special_tokens=True,
                                                        padding="max_length",
                                                        max_length=self.max_token_len)
            input_ids = tokenized_data["input_ids"]
            attention_mask = tokenized_data["attention_mask"]
            self.test_input_ids_batch.append(input_ids)
            self.test_attention_mask_batch.append(attention_mask)
            self.test_labels_batch.append(0)

    def add_ood_data(self, path):
        with open(path) as file:
            lines = file.readlines()
        for line in lines:
            try:
                group_id, class_id, sent = line.strip().split(',', 2)
            except ValueError:
                continue
            tokenized_data = self.tokenizer.encode_plus(sent,
                                                        add_special_tokens=True,
                                                        padding="max_length",
                                                        max_length=self.max_token_len)
            input_ids = tokenized_data["input_ids"]
            attention_mask = tokenized_data["attention_mask"]
            self.test_input_ids_batch.append(input_ids)
            self.test_attention_mask_batch.append(attention_mask)
            self.test_labels_batch.append(1)


if __name__ == "__main__":
    data_loader = TestDataLoader(plm_for_tokenize="bert-base-cased", max_token_len=128)

    data_loader.add_ind_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/CLINIC/demo_clinic.csv")
    data_loader.add_ood_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/M-CID/demo_mcid.csv")

    print("--finished--")