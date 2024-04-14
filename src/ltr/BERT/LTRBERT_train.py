import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
import Bio.SeqIO as SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from huggingface_hub import interpreter_login
import wandb
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from transformers import Trainer, TrainingArguments
import wandb
from helper_functions import *
import math
import sys
# Load the tokenizer and model
kmer = "6"

tokenizer = BertTokenizer.from_pretrained(f'zhihan1996/DNA_bert_{kmer}')
model = BertForSequenceClassification.from_pretrained(f'zhihan1996/DNA_bert_{kmer}', num_labels=2)

# Set model to GPU if possible
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
model.to(device)

def Kmers_funct(seq, size=6, stride=1):
   return [seq[x:x+size].upper() for x in range(0, len(seq) - size + 2, stride)]
def tok_func(x, size=6, stride=1): return " ".join(Kmers_funct(x.replace("N", ""), size=size, stride=stride))

MAX_LEN=512
STRIDE_SIZE=1

LTRs = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]

n_sequences = len(LTRs)

generated, genomic, markov = int(n_sequences*0.15), int(n_sequences*0.6), int(n_sequences*0.25)

genomic_non_LTRs = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/non_LTRs_training_genomic_extracts.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
if genomic < len(genomic_non_LTRs):
    genomic_non_LTRs = random.sample(genomic_non_LTRs, genomic)
generated_non_LTRs = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/non_LTRs_training_generated.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
if generated < len(generated_non_LTRs):
    generated_non_LTRs = random.sample(generated_non_LTRs, generated)
markov_non_LTRs = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/non_LTRs_training_markovChain.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
if markov < len(markov_non_LTRs):
    markov_non_LTRs = random.sample(markov_non_LTRs, markov)
non_LTRs = genomic_non_LTRs + generated_non_LTRs + markov_non_LTRs

records = non_LTRs + LTRs

labels = [0]*len(non_LTRs) + [1]*len(LTRs)

# Split into train and test
trainX, valX, trainY, valY = train_test_split(records, labels, test_size=0.1, random_state=42)

# transform records to sequences and extract info for later analysis
trainX = [str(rec.seq) for rec in trainX]
validation_IDs = [rec.id for rec in valX]
valX = [str(rec.seq) for rec in valX]

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


wandb.login()

wandb.init("LTRBERT_LTR_final_train")

interpreter_login()
print("Preprocessing data...")
train_dataset = Dataset(tokenizer([tok_func(x, int(kmer), STRIDE_SIZE) for x in trainX], padding=True, truncation=True, max_length=512), trainY)
val_dataset = Dataset(tokenizer([tok_func(x, int(kmer), STRIDE_SIZE) for x in valX], padding=True, truncation=True, max_length=512), valY)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

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
    run_name="LTRBERT_LTR_train",
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
print("Training model...")
trainer.train()

# save the trained model to the huggingface hub
trainer.save_model("LTRBERT_LTR_classifier_512")
#trainer.push_to_hub("LTRBERT_LTR_classifier_512")

wandb.finish()
# Tokenize test data
test_trainer = Trainer(model) # Make prediction
raw_pred, _, _ = test_trainer.predict(val_dataset) # Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

pd.DataFrame({"ID": validation_IDs, "prediction": y_pred, "actual": valY}).to_csv("LTRBERT_validation_predictions.csv", index=False)

