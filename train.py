# =======================
# mandatory coding task 2
# =======================

import torch.nn as nn

from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels, _ in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)