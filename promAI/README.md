# promAI

AI-powered system for evaluating **Passive Range of Motion (PROM) exercises** in physical therapy education.  
The project uses **pose estimation preprocessing** (done by a separate pipeline) and a **GRU classifier** for detecting whether a given exercise performance is **correct** or **incorrect**.

---

## Project Structure

```
promAI/
│
├── scripts/
│   ├── clean_data.py           # Standardizes raw dataset folder and file names
│   ├── split_data.py           # Splits raw data into train/val/test and generates manifests
│   ├── report_splits.py        # Summarizes split statistics and class balance
│   └── report_data.py          # Checks for NaNs, shape consistency, and constant features in splits
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── loader.py
│   ├── evaluate.py
│   ├── models/
│   │     └── gru_classifier.py
│   └── utils/
│         └── checkpoint.py
│
├── data/
│   └── processed/
│       └── dataset_v1/
│       │  ├── elbow_extension/
│       │  │    ├── train/
│       │  │    ├── val/
│       │  │    └── test/
│       │  └── shoulder_flexion/
│       │       ├── train/
│       │       ├── val/
│       │       └── test/
│       └── dataset_v2/
│           ├── elbow_extension/
│           │    ├── train/
│           │    ├── val/
│           │    └── test/
│           └── shoulder_flexion/
│                ├── train/
│                ├── val/
│                └── test/
├── checkpoints/
│   ├── elbow_extension/
│   └── shoulder_flexion/
├── results/
│   ├── elbow_extension/
│   └── shoulder_flexion/
└── README.md
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone git@github.com:koperario/promAI.git
   cd promAI
   ```
2. **Create a conda environment:**
   ```sh
   conda create -n promAI python=3.10 -y
   conda activate promAI
   ```
3. **Install dependencies:**
   ```sh
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install numpy pandas scikit-learn matplotlib
   ```
4. **IMPORTANT: Sort Original/Raw Dataset:**
   
   Your raw dataset versions should be organized as follows.

   Example structure for `dataset_v2`:

   ```
   Thesis_Dataset/
   ├── dataset_v1/
   |    └── ...
   ├── dataset_v2/
   |    ├── elbow_extension/
   |    │   ├── correct/
   |    │   │   ├── elbow_extension_correct_001.csv
   |    │   │   ├── elbow_extension_correct_002.csv
   |    │   │   └── ...
   |    │   ├── incorrect/
   |    │   │   ├── elbow_extension_incorrect_001.csv
   |    │   │   ├── elbow_extension_incorrect_002.csv
   |    │   │   └── ...
   |    │   └── subtle_incorrect/
   |    │       ├── elbow_extension_subtle_incorrect_001.csv
   |    │       ├── elbow_extension_subtle_incorrect_002.csv
   |    │       └── ...
   |    └── shoulder_flexion/
   |        ├── correct/
   |        │   ├── shoulder_flexion_correct_001.csv
   |        │   ├── shoulder_flexion_correct_002.csv
   |        │   └── ...
   |        ├── incorrect/
   |        │   ├── shoulder_flexion_incorrect_001.csv
   |        │   ├── shoulder_flexion_incorrect_002.csv
   |        │   └── ...
   |        └── subtle_incorrect/
   |            ├── shoulder_flexion_subtle_incorrect_001.csv
   |            ├── shoulder_flexion_subtle_incorrect_002.csv
   |            └── ...
   └── dataset_v3/
      └── ...
   ```

- You can use have other data dataset versions with any name (e.g., `dataset_v1`, `dataset_v2`, `dataset_v3`, etc.). BUT please stick to current naming convention.
- Each exercise (`elbow_extension`, `shoulder_flexion`) has three subfolders: `correct`, `incorrect`, and `subtle_incorrect`.
- Each subfolder contains CSV files for that exercise and correctness type.

**Make sure your folder and file names follow this structure for smooth processing!**

---

## Workflow

### **Step 1: Data Cleaning**
Standardize raw dataset folder and file names:
```sh
python scripts/clean_data.py --dataset_version dataset_v2
```

### **2. Data Splitting**
- Reflect here where you saved the dataset: split_data.py
Line to Edit:
raw_root = os.path.expanduser("~/Downloads/Thesis_Dataset")

- Run `scripts/split_data.py` to split raw data into train/val/test sets (default: 70/20/10) and generate manifest files.
- Splits are balanced and mutually exclusive.
```sh
python scripts/split_data.py --dataset_version dataset_v2 --exercise elbow_extension
python scripts/split_data.py --dataset_version dataset_v2 --exercise shoulder_flexion
```
### **3. Data Quality Checks**
Summarize split statistics, class balance, and check for duplicates:
```sh
python scripts/report_splits.py --dataset_version dataset_v2
```
Check for NaNs, shape consistency, and constant features in splits:
```sh
python scripts/report_data.py --dataset_version dataset_v2
```

### **4. Training**
- Train the GRU classifier for a specific exercise and dataset version:
  ```sh
  python -m src.train --exercise elbow_extension --dataset_version dataset_v2
  python -m src.train --exercise shoulder_flexion --dataset_version dataset_v2
  ```
- Model checkpoints are saved every few epochs in `checkpoints/<exercise>/`.

### **5. Evaluation**
- By default, it will run the model with the best epoch results.
- Evaluate the trained model on any split (train/val/test) for any exercise and dataset version:
  ```sh
  python -m src.evaluate --exercise elbow_extension --split test --dataset_version dataset_v2
  python -m src.evaluate --exercise shoulder_flexion --split test --dataset_version dataset_v2
  ```
- Prints classification report and confusion matrix; saves results in `results/<exercise>/`.

- Optional: To run a specific model based on other epoch results, run a similar command below.
  Example for running Epoch 10 from dataset_v2 of Elbow Extension:
  ```sh
  python -m src.evaluate --exercise elbow_extension --split test --dataset_version dataset_v2 --checkpoint 250917_230345_1_10.pth
  ```

### **6. Model Analysis & Tuning**
- If needed, tune hyperparameters, modify model architecture, or experiment with feature selection (e.g., remove constant features).
- Retrain and re-evaluate as needed.

---

## Notes
- **Data:** Not included in this repository. Preprocessed CSVs will be placed in `data/processed/`.
- **Model Checkpoints:** Saved in `checkpoints/` (ignored by Git).
- **Manage Model Checkpoints:** *Backup or delete* the .pth files when running another training.
- **Results:** Evaluation metrics are saved in `results/`.

## Troubleshooting
- If you change your splits or fix label issues, **delete old checkpoints** before retraining.
- Always check your manifests for correct label assignment.
- Use the report scripts to verify data integrity before training.
