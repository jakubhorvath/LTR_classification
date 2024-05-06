import torch
from transformers import BertTokenizer, BertForSequenceClassification
import Bio.SeqIO as SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from huggingface_hub import interpreter_login
import wandb
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from transformers import Trainer, TrainingArguments
import wandb
import sys
from utils.BERT_utils import * 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seq_file', help='Path to sequence file')
args = parser.parse_args()
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = balanced_accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


MAX_LEN=510
STRIDE_SIZE=1

kmer = "6"

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(f'zhihan1996/DNA_bert_{kmer}')
model = BertForSequenceClassification.from_pretrained(f'zhihan1996/DNA_bert_{kmer}', num_labels=2)

if torch.cuda.is_available():    

    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model.to(device)

records  = [rec for rec in SeqIO.parse(args.seq_file, "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0 and rec.description.split()[3] != "NAN"]


labels = [rec.description.split()[3] for rec in records]
le = LabelEncoder()
labels = list(le.fit_transform(labels))

# Split into train and test
trainX, valX, trainY, valY = train_test_split(records, labels, test_size=0.1, random_state=42)


# transform records to sequences and extract info for later analysis
trainX = [str(rec.seq) for rec in trainX]
validation_IDs = [rec.id for rec in valX]
valX = [str(rec.seq) for rec in valX]




wandb.login()

wandb.init(project="BERT_superfamily_training")



interpreter_login()
print("Encoding sequences")
train_dataset = Dataset(tokenizer([tok_func(x, int(kmer), STRIDE_SIZE) for x in trainX], padding=True, truncation=True, max_length=512), trainY)
val_dataset = Dataset(tokenizer([tok_func(x, int(kmer), STRIDE_SIZE) for x in valX], padding=True, truncation=True, max_length=512), valY)

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    seed=0,
    load_best_model_at_end=True,
    run_name="LTRBERT_superfamily_kmer_test_" + kmer 
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

# save the trained model to the huggingface hub
trainer.save_model("LTRBERT_superfamily_512")
trainer.push_to_hub()

wandb.finish()
# Tokenize test data
test_trainer = Trainer(model) # Make prediction
raw_pred, _, _ = test_trainer.predict(val_dataset) # Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

pd.DataFrame({"ID": validation_IDs, "prediction": y_pred, "actual": valY}).to_csv("LTRBERT_family_validation_predictions.csv", index=False)

import pickle
pickle.dump(model, open("LTRBERT_family_classifier.b", "wb+"))
