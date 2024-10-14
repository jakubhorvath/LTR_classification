import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from Bio import SeqIO
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

import tqdm
import pickle
from BERT_model import LTRBERT
from CNN_model import Conv1DModel, CNN_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from torch.utils.data import DataLoader
import tqdm 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, matthews_corrcoef
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
test_size = 0.001
sys.path.append("/data/xhorvat9/LTR_classification/src")
sys.path.append("../../")
from utils.BERT_utils import tok_func, Dataset
from utils.CNN_utils import onehote
import sys

def get_embeddings(bert_model, long_sequences):
    min_len = 510
    window_size = min_len
    stride = min_len//3 # ~ 1/3 of window size

    outputs = []
    sequences = []
    counter = 0
    for seq in long_sequences:
        seq_windows = []
        for i in range(0, len(seq), stride):
            start = i
            end = i + window_size

            if end > len(seq):
                end = len(seq)
            seq_windows.append(seq[start:end])
        sequences.append(seq_windows)
    
    # Get the embeddings for the last layer on the split up sequences
    layer_index = -1  # Index of the last layer
    token_index = 0   # Index of the token you're interested in
    model_embeddings = []
    bert_model.eval()
    for s in tqdm.tqdm(sequences):
        counter += 1
    # Tokenize the sequences
        tokenized_segment = tokenizer([tok_func(sequence_segment) for sequence_segment in s], padding=True, truncation=True, max_length=min_len+2, return_tensors="pt")
        tokenized_segment.to("cuda")
        with torch.no_grad():
            # Run model on segments
            outputs = bert_model(**tokenized_segment, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            embeddings = hidden_states[layer_index][:, token_index, :]
            embeddings = embeddings.to("cpu")
            model_embeddings.append(embeddings)

    averaged = []
    for emb in tqdm.tqdm(model_embeddings):
        averaged.append((sum(emb)/len(emb)).numpy())
    averaged = np.array(averaged)
    return averaged

def pad_sequences(array, max_len):
    if len(array) < max_len:
        padding = [[0,0,0,0]] * (max_len - len(array))
        padded = array + padding
        return padded
    else:
        return array

def compute_metrics(p, NO_ARGMAX=False):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    labels = np.argmax(labels, axis=1)

    accuracy = balanced_accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    mcc = matthews_corrcoef(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "MCC": mcc}
    
class CNN_dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return pad_sequences(self.data[idx], 4000), self.target[idx]

class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(128 * 383, 128)  # Adjusting input size for the Dense layer
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 13)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.ReLU()(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
        
def select_short_sequences(X, y, max_len=510):
    short_sequences = []
    short_labels = []
    long_sequences = []
    long_labels = []
    for i, seq in enumerate(X):
        if len(seq) <= max_len:
            short_sequences.append(seq)
            short_labels.append(y[i])
        else:
            long_sequences.append(seq)
            long_labels.append(y[i])
    return np.array(short_sequences), np.array(short_labels), np.array(long_sequences), np.array(long_labels)

class BERT_dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/LTR_classification_data/Sequence_files/train_LTRs.fasta", "fasta") if rec.description.split()[3] != "NAN"]

LTR_motifs = pd.read_csv("/data/xhorvat9/LTR_classification_data/TFBS/LTR_train_motifCounts.csv", sep="\t").set_index("ID")

# LTR ordering is identical to its motif representation
LTR_sequence_df = pd.DataFrame({"sequence": [str(rec.seq) for rec in LTRs], "ID": [rec.id for rec in LTRs], "label": [rec.description.split()[4] for rec in LTRs]})
LTR_sequence_df.set_index("ID", inplace=True)
LTR_sequence_df = LTR_sequence_df.merge(LTR_motifs, on="ID")[["sequence", "label"]]
LTR_sequence_df = LTR_sequence_df[LTR_sequence_df["label"].isin(list(LTR_sequence_df["label"].value_counts().iloc[:13].index))]
X_motifs = LTR_sequence_df.merge(LTR_motifs, on="ID").drop(["sequence", "label"], axis= 1)
print("Indices for LTR sequences and motifs are identical: ", all(X_motifs.index == LTR_sequence_df.index))



X = np.array(LTR_sequence_df["sequence"].tolist())
le = LabelEncoder()
y = le.fit_transform(LTR_sequence_df["label"])


X_indices, _, _, _ = train_test_split([i for i in range(len(X))], y, test_size=test_size, shuffle=True, random_state=42)

X = X[X_indices]
y = y[X_indices]

from sklearn.preprocessing import OneHotEncoder
y_ohe = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

X_motifs = X_motifs.iloc[X_indices, ]

# TF-IDF transformation of motifs
tfidf = TfidfTransformer()
X_motifs = tfidf.fit_transform(X_motifs).toarray()


# Preprocess the data 
tokenizer = transformers.BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')

CNN_input_size = 4000

#X_OHE = [onehote(x) for x in X]
#X_OHE = np.array(X_OHE, dtype="object")

kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
kf.get_n_splits(X, y)
splits = kf.split(X, y)

old_stdout = sys.stdout
log_file = open("training_out.log","w+")

sys.stdout = log_file

#np.save("splits.npy", np.array(splits, dtype="object"))

for i, (train_index, test_index) in enumerate(splits):

    # Train the BERT model

    bert_model = transformers.BertForSequenceClassification.from_pretrained('zhihan1996/DNA_bert_6', num_labels=13)

    # Tokenize the short sequences
    BERT_train_X, train_y = X[train_index], y[train_index]
    BERT_train_X_short, train_y_short, BERT_train_X_long, train_y_long = select_short_sequences(BERT_train_X, y_ohe[train_index])
    
    BERT_test_X_short, test_y_short, BERT_test_X_long, test_y_long = select_short_sequences(X[test_index], y_ohe[test_index])

    train_dataset = Dataset(tokenizer([tok_func(x) for x in BERT_train_X_short], padding=True, truncation=True, max_length=512), train_y_short)
    val_dataset = Dataset(tokenizer([tok_func(x) for x in BERT_test_X_short], padding=True, truncation=True, max_length=512), test_y_short)

    # Train BERT on short sequences 
    args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    seed=0,
    load_best_model_at_end=True,
    run_name="LTRBERT_LTR_train",
    push_to_hub=True,)

    trainer = Trainer(
        model=bert_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()        

    # Run model on segments of long sequences

    averaged = get_embeddings(bert_model, BERT_train_X_long)

    pickle.dump(averaged, open(f"averaged_{i}.pkl", "wb"))

    emb_model = EmbeddingNet().cuda()
    criterion = torch.nn.CrossEntropyLoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(emb_model.parameters())  # Adam optimizer

    batch_size = 32
    for epoch in range(3):
        for batch_index in range(0, len(averaged), batch_size):
            batch_X = torch.tensor(averaged[batch_index:batch_index+batch_size, :], dtype=torch.float).unsqueeze(1).cuda()
            batch_Y = torch.tensor(train_y_long[batch_index:batch_index+batch_size], dtype=torch.float)

            outputs = emb_model(batch_X)  # PyTorch expects channels first, so we transpose
            optimizer.zero_grad()
            loss = criterion(outputs.cpu(), batch_Y)
            loss.backward()
            optimizer.step()


    # Train the CNN
    OHE_train_X, OHE_test_X  = X_OHE[train_index], X_OHE[test_index]
    CNN_model = Conv1DModel()
    CNN_model = CNN_model.to("cuda")
    # Define the loss function and optimizer
    CNN_criterion = torch.nn.CrossEntropyLoss()  # Binary Cross Entropy Loss
    CNN_optimizer = torch.optim.Adam(CNN_model.parameters(), lr = 0.01 )

    OHE_train_X, OHE_test_X  = X_OHE[train_index], X_OHE[test_index]
    batch_size = 256
    for epoch in range(3):
        for batch_index in range(0, len(OHE_train_X), batch_size):
            batch_X = OHE_train_X[batch_index:batch_index+batch_size]
            batch_Y = torch.tensor(y_ohe[train_index][batch_index:batch_index+batch_size], dtype=torch.float)
    
            padded_batch_X = torch.tensor(np.array([pad_sequences(x.tolist(), 4000) for x in batch_X]), dtype=torch.float).permute(0, 2, 1).to("cuda")
            outputs = CNN_model(padded_batch_X)  # PyTorch expects channels first, so we transpose
            #????? should this be before or after running the model 
            
            
            loss = CNN_criterion(outputs, batch_Y.cuda())
            CNN_optimizer.zero_grad()
            loss.backward()
            CNN_optimizer.step()

    
    GBC = GradientBoostingClassifier(max_depth=8, min_samples_leaf=50, n_estimators=400)
    # Train the GBC
    X_motifs_train, X_motifs_test = X_motifs[train_index], X_motifs[test_index]
    GBC.fit(X_motifs_train, train_y)


    # Test the trained classifiers 
    #raw_pred, _, _ = trainer.predict(Dataset(tokenizer([tok_func(x) for x in BERT_test_X_short], padding=True, truncation=True, max_length=512), test_y_short))
    #bert_short_predictions = np.argmax(raw_pred, axis=1)
    bert_short_predictions, _, _ = trainer.predict(Dataset(tokenizer([tok_func(x) for x in BERT_test_X_short], padding=True, truncation=True, max_length=512)))
    emb_model.eval()
    bert_long_predictions = emb_model(torch.tensor(get_embeddings(bert_model, BERT_test_X_long), dtype=torch.float).unsqueeze(1).cuda())
    
    CNN_model.eval()

    padded_test_X = torch.tensor(np.array([pad_sequences(x.tolist(), 4000) for x in OHE_test_X]), dtype=torch.float).permute(0, 2, 1).to("cuda")
    CNN_predictions = CNN_model(padded_test_X)
    
    GBC_predictions = GBC.predict(X_motifs_test)

    # Evaluate predictions
    bert_short_metrics = compute_metrics((bert_short_predictions, test_y_short), True)
    bert_long_metrics = compute_metrics((bert_long_predictions.detach().cpu(), test_y_long), True)

    CNN_metrics = compute_metrics((CNN_predictions.detach().cpu(), y_ohe[test_index]), True)

    GBC_metrics = compute_metrics((GBC_predictions, y[test_index]), True)
    print(f"Split {i}")
    print(f"BERT short sequence metrics: {bert_short_metrics}")
    print(f"BERT long sequence metrics: {bert_long_metrics}")
    print(f"CNN metrics: {CNN_metrics}")
    print(f"GBC metrics: {GBC_metrics}")


sys.stdout = old_stdout

log_file.close()