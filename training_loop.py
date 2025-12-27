# =======================
# mandatory coding task 2
# =======================

from eval import evaluate
from train_epoch import train_one_epoch


def train_model(model, trian_loader, val_loader, optimizer, device, num_epochs):
    history = {
        "train_loss": [],
        "val_acc": [],
        "val_tpr_per_class": []
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, trian_loader, optimizer, device)
        val_acc, val_tpr = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_tpr_per_class"].append(val_tpr)

        print(
            f"Epoch {epoch+1:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    return history, best_state