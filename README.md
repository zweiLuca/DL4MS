# Deep Learning - Mandatory Coding Task
```
Luca Martin     - 1234567
```

## Setup

- run `python -m venv venv` inside the project root folder in the terminal
- start the virtual environment
- run `pip install -r requirements.txt` in the terminal
- unzip the images (e.g. into `./coding_task_data/EuroSAT_RGB` and `./coding_task_data/EuroSAT_MS`)
- go into `./config.yml`
    - Set `project_root` and `dataset_root`

## Structure

1. Coding Task 1 is solved inside `./data_splitter.py`
2. To run the prediction on the test data using the final model, run `./predict.py`
3. To run the reproduction routine using the final model, run `./reproduce.py`

```
project_root/
│
├── coding_task_data/          # Directory containing the RBG and MS dataset
│   ├── EuroSAT_RGB            # Task 2 Dataset
│   └── EuroSAT_MS             # Task 3 Dataset
│
├── datasets/
│   ├── dataloader.py          # Task 2: CustomDataLoader
│   ├── eurosat_dataset.py     # Task 2: RGB Dataset class
│   ├── eurosat_ms_dataset.py  # Task 3: MS Dataset class
│   └── transforms.py          # Task 2: train/eval transforms (mild/strong)
│
├── evaluation/
│   ├── eval.py                # Task 2: evaluate() function
│   └── plots.py               # Task 2: plot_validation_accuracy(), plot TPR
│
├── models/
│   ├── model.py               # Task 2: Model class + get_model()
│   ├── feature_extractor.py   # Task 3: Feature extractor with late fusion
│   └── custom_classifier.py   # Task 3: Custom ResNet model with late fusion
│
├── outputs/
│   ├── test_logits.pt         # Task 2: Saved test logits
│   ├── test_paths.pt          # Task 2: Saved test paths
│   ├── test_logits_ms.pt      # Task 3: Saved test logits
│   ├── test_paths_ms.pt       # Task 3: Saved test paths
│   ├── graphs/                # Task 2 & 3: val_acc, val_tpr graphs
│   └── ranking/               # Task 2 & 3: top/bottom images
│
├── splits/                    # Task 2 & 3: Splits for RGB and MS
│
├── training/
│   ├── train_epoch.py         # Task 2: train_one_epoch funcion
│   └── training_loop.py       # Task 2: train_model
│
├── utils/
│   ├── config.py              # Task 2: load_config() function
│   └── set_seed.py            # Task 2: reproducability for torch, np, random
│
├── requirements.txt          
├── config.yml                 # all paths, seeds, hyperparameters
├── data_splitter.py           # Task 1: creates train/val/test splits
├── predict.py                 # Task 2 & 3: test set predictions
├── ranking.py                 # Task 2 & 3: top-k / bottom-k per label
├── reproduce.py               # Task 2 & 3: reproduction routine
└── train.py                   # Task 2 & 3: training script
```