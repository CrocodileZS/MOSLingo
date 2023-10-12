from dataloader.MOSLingoDataLoader import MOSLingoDataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import time
from utils.log_utils import get_logger
from transformers import BertModel
import torch.nn.functional as F

from utils.loss_utils import calc_group_softmax_loss

# configurations
batch_size = 5
plm_for_tokenize = "bert-base-cased"
num_group = 4

def main():
    # load the dataset
    data_loader = MOSLingoDataLoader(plm_for_tokenize=plm_for_tokenize,
                                     num_group=num_group,
                                     max_token_len=128)

    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/CLINIC/demo_clinic.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/M-CID/demo_mcid.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/HWU/demo_hwu.csv")
    data_loader.add_group_data("/Users/zhouyuyang/PycharmProjects/MOSLingo/dataset/Snips/demo_snips.csv")

    input_ids = torch.tensor([i for i in data_loader.input_ids_batch], dtype=torch.long)
    attention_mask = torch.tensor([i for i in data_loader.attention_mask_batch], dtype=torch.long)
    label_ids = torch.tensor([i for i in data_loader.labels_grouped_batch], dtype=torch.long)
    group_slice = torch.tensor(data_loader.group_slice)
    train_dataset = TensorDataset(input_ids, attention_mask, label_ids)

    # save group slice for test
    torch.save(group_slice, "./group_slice.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # fine-tune the model
    model = BertModel.from_pretrained("bert-base-cased")
    model.to(device)

    # training parameters
    EPOCHS = 5
    learning_rate = 5e-6
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    logger = get_logger(f'./train_{time.time()}.log')

    logger.info('Batch Size: ' + str(batch_size) + '\tLearning Rate: ' + str(learning_rate) + '\n')

    # start training
    for Epoch in range(1, EPOCHS + 1):

        # TODO: Resume fine-tuning if we find a saved model.

        model.train()

        # set a linear for classification
        linear_input_size = model.embeddings.position_embeddings.embedding_dim  # last layer dimension
        linear_output_size = data_loader.group_slice[-1][1] + 1
        linear = nn.Linear(in_features=linear_input_size, out_features=linear_output_size)

        for batch_index, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # forward
            x = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output

            logits = linear(x)
            pred_prob = F.softmax(logits)

            # calculate loss
            loss = calc_group_softmax_loss(criterion, pred_prob, labels, group_slice)

            logger.info('Epoch: %d ｜ Train: | Batch: %d / %d | Loss: %f' %
                  (Epoch, batch_index + 1, len(train_dataloader), loss)
            )

            model.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()

            # TODO: add model.evel()

        if Epoch % 2 == 0:
            saved_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'linear': linear.state_dict()
            }
            torch.save(saved_dict, f'./grouped_model{time.time()}.pth.tar')


if __name__ == '__main__':
    main()
