# =======================
# mandatory coding task 2
# =======================

import matplotlib.pyplot as plt


def plot_validation_accuracy(history, save_path):
    epochs = range(1, len(history["val_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["val_acc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy per Epoch")
    plt.xticks(epochs)
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_validation_tpr(history, class_names, save_path):
    epochs = range(1, len(history["val_tpr_per_class"]) + 1)

    plt.figure(figsize=(10, 6))

    class_ids = sorted(history["val_tpr_per_class"][0].keys())

    for cls_id in class_ids:
        tpr_values = [
            epoch_tpr.get(cls_id, 0.0)
            for epoch_tpr in history["val_tpr_per_class"]
        ]
        plt.plot(
            epochs,
            tpr_values,
            marker="o",
            label=class_names[cls_id]
        )

    plt.xlabel("Epoch")
    plt.ylabel("TPR")
    plt.title("Validation TPR per Class")
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="small",
        ncol=1
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()