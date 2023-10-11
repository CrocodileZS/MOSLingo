def calc_group_softmax_loss(criterion, pred_prob, labels, group_slices):
    num_groups = group_slices.shape[0]
    loss = 0
    for i in range(num_groups):
        group_logit = pred_prob[:, group_slices[i][0]: group_slices[i][1]+1]
        group_label = labels[:, i]
        loss += criterion(group_logit, group_label)
    return loss