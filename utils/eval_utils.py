import torch
import numpy as np
import ood_metrics
import torch.nn.functional as F


def run_eval(logger, pred_prob, group_slice, labels):
    logger.info("Running test...")
    # logger.flush()

    final_max_score = get_others_probs(pred_prob, group_slice)
    auroc, fpr95, detection_error = get_ood_metrics_without_threshold(final_max_score, labels)

    logger.info('============Results for MOS============')
    logger.info('AUROC: {}'.format(auroc))
    logger.info('FPR95: {}'.format(fpr95))
    logger.info('DETECTION ERROR: {}'.format(detection_error))

    # logger.flush()


def run_eval_ungrouped(logger, pred_prob, group_slice, labels):
    logger.info("Running test...")
    # logger.flush()

    other_probs = get_others_probs_ungrouped(pred_prob)
    auroc, fpr95, detection_error = get_ood_metrics_without_threshold_ungrouped(other_probs, labels)

    logger.info('============Results for MOS============')
    logger.info('AUROC: {}'.format(auroc))
    logger.info('FPR95: {}'.format(fpr95))
    logger.info('DETECTION ERROR: {}'.format(detection_error))

    # logger.flush()


def get_others_probs(pred_prob, group_slice):
    num_groups = group_slice.shape[0]

    all_group_ood_score_MOS = []

    for i in range(num_groups):
        group_logit = pred_prob[:, group_slice[i][0]: group_slice[i][1] + 1]
        group_softmax = F.softmax(group_logit, dim=1)
        others_prob = group_softmax[:, 0]
        # others_prob = pred_prob[:, group_slice[i][0]]
        all_group_ood_score_MOS.append(others_prob)

    all_group_ood_score_MOS = torch.stack(all_group_ood_score_MOS, dim=1)
    final_max_score_MOS, _ = torch.max(all_group_ood_score_MOS, dim=1)
    return final_max_score_MOS.data.cpu().numpy()


def get_others_probs_ungrouped(pred_prob):
    group_softmax = F.softmax(pred_prob, dim=1)
    others_prob = group_softmax[:, 0]
    return others_prob.cpu().numpy()


def pred_ood(final_max_score, threshold):
    threshold_matrix = np.full(final_max_score.shape, threshold, dtype=float)
    res = np.ceil(np.subtract(final_max_score, threshold_matrix))
    return res


def get_ood_metrics_without_threshold(final_max_score, labels):
    metrics = ood_metrics.calc_metrics(final_max_score.tolist(), labels.tolist())
    return metrics['auroc'], metrics['fpr_at_95_tpr'], metrics['detection_error']


def get_ood_metrics_without_threshold_ungrouped(others_prob, labels):
    metrics = ood_metrics.calc_metrics(others_prob.tolist(), labels.tolist())
    return metrics['auroc'], metrics['fpr_at_95_tpr'], metrics['detection_error']
