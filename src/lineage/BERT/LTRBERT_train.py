import torch
from transformers import BertTokenizer, BertForSequenceClassification
import Bio.SeqIO as SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from huggingface_hub import interpreter_login
import wandb
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, accuracy_score
import pandas as pd
from transformers import Trainer, TrainingArguments
import wandb
import sys
import random
from utils import BERT_utils


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    weighted_accuracy = balanced_accuracy_score(labels, pred)

    return {"weighted_accuracy": weighted_accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

MAX_LEN=510
STRIDE_SIZE=1

kmer = "6"
# Load the tokenizer and model
n_classes = 15
tokenizer = BertTokenizer.from_pretrained(f'zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained(f'zhihan1996/DNA_bert_6', num_labels=n_classes)

# Check if a GPU is available and if so, use it
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
model.to(device)

LTRs = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]

n_sequences = len(LTRs)

d = pd.DataFrame({'sequence':[str(rec.seq) for rec in LTRs], 'label':[rec.description.split(" ")[4] for rec in LTRs]})
d = d[~d['label'].str.contains("copia")]
d = d[d["label"].isin(d["label"].value_counts()[:n_classes].index.tolist())]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(d["label"])
# Split into train and test
trainX, valX, trainY, valY = train_test_split(d["sequence"], labels, test_size=0.1, random_state=42)

# Connect to wandb interface
wandb.login()
wandb.init(project="BERT_lineage_training")
interpreter_login()

# Tokenize training and validation data
print("Encoding sequences")
train_dataset = BERT_utils.Dataset(tokenizer([BERT_utils.tok_func(x, int(kmer), STRIDE_SIZE) for x in trainX], padding=True, truncation=True, max_length=512), trainY)
val_dataset = BERT_utils.Dataset(tokenizer([BERT_utils.tok_func(x, int(kmer), STRIDE_SIZE) for x in valX], padding=True, truncation=True, max_length=512), valY)

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    seed=0,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Training model...")
trainer.train()

# save the trained model to the huggingface hub
trainer.save_model("LTRBERT_lineage_512")
trainer.push_to_hub()

wandb.finish()
# Tokenize test data
test_trainer = Trainer(model) # Make prediction
raw_pred, _, _ = test_trainer.predict(val_dataset) # Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

pd.DataFrame({"ID": validation_IDs, "prediction": y_pred, "actual": valY}).to_csv("LTRBERT_lineage_validation_predictions.csv", index=False)