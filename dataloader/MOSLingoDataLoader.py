from transformers import BertTokenizer
import numpy as np


class MOSLingoDataLoader:
    def __init__(self,
                 plm_for_tokenize,
                 num_group,
                 max_token_len):
        self.plm = plm_for_tokenize
        self.tokenizer = BertTokenizer.from_pretrained(self.plm)
        self.num_group = num_group
        self.input_ids_batch = []
        self.attention_mask_batch = []
        self.labels_grouped_batch = []
        self.labels_ungrouped_batch = []
        self.max_token_len = max_token_len
        self.num_classes_each_group_without_others = []
        self.group_slice = []

    def add_group_data(self, path):
        with open(path) as file:
            lines = file.readlines()
        max_class_id = -1
        for line in lines:
            group_id, class_id, sent = line.strip().split(',', 2)
            max_class_id = max(max_class_id, int(class_id))
            tokenized_data = self.tokenizer.encode_plus(sent,
                                                        add_special_tokens=True,
                                                        padding="max_length",
                                                        max_length=self.max_token_len,
                                                        truncation=True)
            input_ids = tokenized_data["input_ids"]
            attention_mask = tokenized_data["attention_mask"]
            label = np.zeros(self.num_group, dtype=int)
            label[int(group_id)] = int(class_id) + 1
            self.input_ids_batch.append(input_ids)
            self.attention_mask_batch.append(attention_mask)
            self.labels_grouped_batch.append(label)
            self.labels_ungrouped_batch.append(
                sum(self.num_classes_each_group_without_others) + int(class_id) + 1
            )

        self.num_classes_each_group_without_others.append(max_class_id + 1)
        if len(self.group_slice) == 0:
            self.group_slice.append([0, max_class_id + 1])
        else:
            last_idx = self.group_slice[-1][1]
            self.group_slice.append([last_idx + 1, last_idx + max_class_id + 2])


if __name__ == "__main__":
    data_loader = MOSLingoDataLoader(plm_for_tokenize="bert-base-cased", num_group=4, max_token_len=50)

    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/CLINIC/G0_CLINIC150_train.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/M-CID/M-CID_train.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/HWU/HWU64_train.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/Snips/G3_SNIPS_train.csv")

    print("end")
