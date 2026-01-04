# =======================
# mandatory coding task 2
# =======================

import torch


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    return_logits: bool = False,
    return_paths: bool = False
    ):
    model.eval()

    correct = 0
    total = 0

    per_class_correct = {}
    per_class_total = {}

    all_logits = []
    all_labels = []
    all_paths = []

    for images, labels,  paths in loader:
        if return_paths:
            all_paths.extend(paths)

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

        if return_logits:
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    acc = correct / total
    tpr_per_class = {
        c: per_class_correct.get(c, 0) / per_class_total[c]
        for c in per_class_total
    }

    output = {
        "accuracy": acc,
        "tpr_per_class": tpr_per_class,
    }

    if return_logits:
        output["logits"] = torch.cat(all_logits)
        output["labels"] = torch.cat(all_labels)

    if return_paths:
        output["paths"] = all_paths

    return output
