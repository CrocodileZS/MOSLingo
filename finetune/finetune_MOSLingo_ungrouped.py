from utils.MOSLingoDataLoader import MOSLingoDataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import time
from utils.log_utils import get_logger
from transformers import BertModel
import torch.nn.functional as F

# configurations
batch_size = 20
plm_for_tokenize = "bert-base-cased"
num_group = 4

def main():
    # load the data
    data_loader = MOSLingoDataLoader(plm_for_tokenize=plm_for_tokenize,
                                     num_group=num_group,
                                     max_token_len=128)

    # must read in group_number order
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/data/CLINIC/demo_clinic.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/data/M-CID/demo_mcid.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/data/HWU/demo_hwu.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/data/Snips/demo_snips.csv")

    input_ids = torch.tensor([i for i in data_loader.input_ids_batch], dtype=torch.long)
    attention_mask = torch.tensor([i for i in data_loader.attention_mask_batch], dtype=torch.long)
    label_ids = torch.tensor([i for i in data_loader.labels_ungrouped_batch], dtype=torch.long)
    train_dataset = TensorDataset(input_ids, attention_mask, label_ids)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # fine-tune the model
    model = BertModel.from_pretrained("bert-base-cased")
    model.to(device)

    # training parameters
    EPOCHS = 10
    learning_rate = 0.00001
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    logger = get_logger(f'./train_{time.time()}.log')

    logger.info('Batch Size: ' + str(batch_size) + '\tLearning Rate: ' + str(learning_rate) + '\n')

    # start training
    for Epoch in range(1, EPOCHS + 1):

        # TODO: Resume fine-tuning if we find a saved model.

        model.train()
        for batch_index, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # forward
            x = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output

            # set a linear for classification
            linear_input_size = model.embeddings.position_embeddings.embedding_dim # last layer dimension
            linear_output_size = data_loader.group_slice[-1][1] + 1
            linear = nn.Linear(in_features=linear_input_size, out_features=linear_output_size)

            logits = linear(x)
            pred_prob = F.softmax(logits)

            # calculate loss
            loss = criterion(pred_prob, labels)

            logger.info('Epoch: %d ｜ Train: | Batch: %d / %d | Loss: %f' %
                  (Epoch, batch_index + 1, len(train_dataloader), loss)
            )

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: add model.evel()

        if Epoch % 5 == 0:
            saved_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(saved_dict, f'./grouped_model{time.time()}.pth.tar')


if __name__ == '__main__':
    main()
