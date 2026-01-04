# Deep Learning - Mandatory Coding Task
```
Luca Martin - 1234567
```

For detailed instructions and the final results, read the file `"Report_Luca_Martin_1234567.pdf"`.

## Structure
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
## Results - RGB
========================
Training mild_aug on cuda with:
- ResNet18 Model on RGB images from coding_task_data/EuroSAT_RGB
- 10000 training images
- 5000 validation images
- 625 batches
- batchsize 16
- 20 epochs
- learning rate 0.0001
- Seed 1234567
========================

Best model:
- Epoch 16
- val_acc 0.9466


========================
Training strong_aug on cuda with:
- ResNet18 Model on RGB images from coding_task_data/EuroSAT_RGB
- 10000 training images
- 5000 validation images
- 625 batches
- batchsize 16
- 20 epochs
- learning rate 0.0001
- Seed 1234567
========================

Best model:
- Epoch 19
- val_acc 0.9200

Final model: mild aug
Validation accuracy: 0.9466
- Accuracy: 0.923
- TPR: {0: 0.928, 1: 0.908, 2: 0.954, 3: 0.9, 4: 0.91, 5: 0.894, 6: 0.864, 7: 0.978, 8: 0.908, 9: 0.986}


## Results - MS
========================
Training mild_aug_ms on cuda with:
- Custom ResNet18 Model on Multispectral images from coding_task_data/EuroSAT_MS
- 10000 training images
- 5000 validation images
- 625 batches
- batchsize 16
- 20 epochs
- learning rate 0.0001
- Seed 1234567
========================

Best model:
- Epoch 6
- val_acc 0.9484

========================
Training strong_aug_ms on cuda with:
- Custom ResNet18 Model on Multispectral images from coding_task_data/EuroSAT_MS
- 10000 training images
- 5000 validation images
- 625 batches
- batchsize 16
- 20 epochs
- learning rate 0.0001
- Seed 1234567
========================

Best model:
- Epoch 5
- val_acc 0.9508

Final model: strong aug
Validation accuracy: 0.9466
- Accuracy: 0.933
- TPR: {0: 0.918, 1: 1.0, 2: 0.942, 3: 0.918, 4: 0.982, 5: 0.934, 6: 0.748, 7: 0.906, 8: 0.982, 9: 1.0}