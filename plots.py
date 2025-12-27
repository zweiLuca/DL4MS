# =======================
# mandatory coding task 2
# =======================

import matplotlib.pyplot as plt


def plot_validation_accuracy(histories, labels, save_path):
    for hist, label in zip(histories, labels):
        plt.plot(hist["val_acc"], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()