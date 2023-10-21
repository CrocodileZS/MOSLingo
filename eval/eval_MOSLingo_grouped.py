import sys
sys.path.append("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo")

from model.MOSLingoModel import MosLingoModel
import torch
from dataloader.EvalDataLoader import TestDataLoader
from torch.utils.data import TensorDataset, DataLoader
from utils.eval_utils import run_eval
import time
from utils.log_utils import get_logger

# configurations
plm_for_tokenize = "bert-base-cased"
batch_size = 5

def main():
    test_data_loader = TestDataLoader(plm_for_tokenize=plm_for_tokenize,
                                 max_token_len=128)
    test_data_loader.add_ood_data("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo/dataset/BANKING/G4_BANKING77_test.csv")
    test_data_loader.add_ind_data("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo/dataset/CLINIC/G0_CLINIC150_test.csv")
    test_data_loader.add_ind_data("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo/dataset/HWU/HWU64_test.csv")
    test_data_loader.add_ind_data("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo/dataset/M-CID/M-CID_test.csv")
    test_data_loader.add_ind_data("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo/dataset/Snips/G3_SNIPS_test.csv")
    test_data_loader.add_ood_data("/gpfs/home6/scur0843/JHS_notebooks/directory to extract/MOSLingo/dataset/StackOverflow/stackoverflow_test.csv")

    input_ids = torch.tensor([i for i in test_data_loader.test_input_ids_batch], dtype=torch.long)
    attention_mask = torch.tensor([i for i in test_data_loader.test_attention_mask_batch], dtype=torch.long)
    label_ids = torch.tensor([i for i in test_data_loader.test_labels_batch], dtype=torch.long)
    test_dataset = TensorDataset(input_ids, attention_mask, label_ids)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    group_slice = torch.load("../finetune/group_slice.pt")

    model = MosLingoModel(plm="bert-base-cased", group_slice=group_slice)
    model.to(device)

    checkpoint = torch.load('../finetune/grouped_model1697678128.6396594.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    logger = get_logger(f'./eval_{time.time()}.log')
    pred_prob = []
    y_true = []

    for batch_index, batch in enumerate(test_dataloader):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # forward
            pred_prob.append(model.forward(input_ids, attention_mask).to('cpu'))
            y_true.append(labels)

            logger.info('Test | Batch: %d / %d' %
                        (batch_index + 1, len(test_dataloader))
                        )

    # pred
    run_eval(logger, torch.cat(pred_prob, dim=0), group_slice, torch.cat(y_true, dim=0))


if __name__ == '__main__':
    main()