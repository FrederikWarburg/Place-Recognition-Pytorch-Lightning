from pytorch_metric_learning import losses, distances


def configure_metric_loss(loss, distance, margin):

    if distance == 'dot':
        dist = distances.DotProductSimilarity()
    elif distance == 'euclidean':
        dist = distances.LpDistance(p=2, normalize_embeddings = True, power=2)

    if loss == 'triplet':
        criterion = losses.TripletMarginLoss(margin=margin, distance=dist)
    else:
        pos_margin = margin if distance == 'dot' else 0
        neg_margin = 0 if distance == 'dot' else margin

        criterion = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin = neg_margin, distance=dist)

    return criterion
