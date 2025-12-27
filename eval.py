
# =======================
# mandatory coding task 2
# =======================

import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    for images, labels,  _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        for l, p in zip(labels, preds):
            l = l.item()
            p = p.item()

            per_class_total[l] = per_class_total.get(l, 0) + 1
            if l == p:
                per_class_correct[l] = per_class_correct.get(l, 0) + 1

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    tpr_per_class = {
        c: per_class_correct.get(c, 0) / per_class_total[c]
        for c in per_class_total
    }

    return acc, tpr_per_class