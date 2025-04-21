# Multi-Task Emotion & Clause Classification - Group 40 NLP

This project performs multi-task classification on dialogue datasets, predicting **emotion** and **clause** labels for each utterance using two separate transformer models. It includes preprocessing, oversampling for class imbalance, individual model training with early stopping, and evaluation metrics like classification reports.

---

## 🔧 Setup Instructions

### 📦 Required Libraries

Install the necessary Python packages using pip:

```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn
```

You might also need `datasets` and `tqdm`:

```bash
pip install datasets tqdm
```

---

## 📁 Dataset

The code uses dialogue data from the [`RECCON`](https://github.com/declare-lab/RECCON) project.

**Download Step:**
```bash
git clone https://github.com/declare-lab/RECCON.git
```

**Original File Paths Used:**
- `RECCON/data/original_annotation/dailydialog_valid.json`
- Original Files can be used from Dataset Diectly 
---

## 🧼 Preprocessing

### Input:
Raw JSON with multiple keys per utterance.

### Output:
A cleaned version of the dataset with only:
```json
["turn", "speaker", "utterance"]
```

## 🧠 Model Training

- **Two Transformers**: One for emotion classification, one for clause classification.
- **Early Stopping**: Based on validation accuracy.
- **Oversampling**: Handled for class imbalance.
- **Saved Models**: Each model is saved separately after training.

---

## 📊 Evaluation

Classification reports are generated for both tasks using `sklearn.metrics.classification_report`.

---

## 📂 File Generation Notes

- You **can reuse** the same preprocessing code with **different dataset filenames** (e.g., change `"dailydialog_valid.json"` to `"dailydialog_train.json"`).
- All intermediary or cleaned JSONs are **regenerable** by rerunning those cells.
- Model checkpoints are saved individually — so emotion and clause models are **independent**.
- Use Files Directly From - https://kaggle.com/datasets/9c7ecdb36c8edbdd7af6fe2b923afaa6fabd1d5684dc0688aad2407225e95547
---

## 🚀 Usage Flow Summary

1. Clone RECCON & prepare dataset.
2. Run the code cell for desired file.
3. Train each model (emotion + clause).
4. Evaluate using provided metrics.
