import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
from transformers import Trainer
import random
import tqdm
import pandas as pd
from utils import BERT_utils
import Bio.SeqIO as SeqIO
from sklearn.model_selection import train_test_split

# Load the tokenizer and model
n_classes = 15
tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained('./LTRBERT_lineage_512', num_labels=n_classes)

# Check if a GPU is available and if so, use it
if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model.to(device)

max_len = 512

# Load the LTR sequences
LTRs  = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta") if len(rec.seq) >= max_len and len(rec.seq) > 0]
d = pd.DataFrame({'sequence':[str(rec.seq) for rec in LTRs], 'label':[rec.description.split(" ")[4] for rec in LTRs], 'seq_id':[rec.id for rec in LTRs]})

# Encode the labels
label_encoder = pickle.load(open("/data/xhorvat9/ltr_bert/NewClassifiers/Lineage/label_encoder.b", "rb"))
d = d[~d['label'].str.contains("copia")]
d = d[d["label"].isin(label_encoder.classes_)]
labels = list(label_encoder.transform(d["label"]))

long_sequences = d["sequence"].tolist()

window_size = max_len
stride = max_len//3 # ~ 1/3 of window size

outputs = []
sequences = []
# Cut sequences into windows
for seq in long_sequences:
  seq_windows = []
  for i in range(0, len(seq), stride):
      start = i
      end = i + window_size

      if end > len(seq):
        end = len(seq)
      seq_windows.append(seq[start:end])
  sequences.append(seq_windows)
  
# Get the embeddings for the last layer and a specific token (e.g., the first token)
layer_index = -1  # Index of the last layer
token_index = 0   # Index of the token you're interested in
model_embeddings = []
model.eval()
for s in tqdm.tqdm(sequences):
  tokenized_segment = tokenizer([BERT_utils.tok_func(sequence_segment) for sequence_segment in s], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
  tokenized_segment.to(device)
  with torch.no_grad():
      outputs = model(**tokenized_segment, output_hidden_states=True)
      hidden_states = outputs.hidden_states
      embeddings = hidden_states[layer_index][:, token_index, :]
      embeddings = embeddings.to("cpu")
      model_embeddings.append(embeddings)

averaged = []
for emb in tqdm.tqdm(model_embeddings):
    averaged.append((sum(emb)/len(emb)).numpy())
averaged = np.array(averaged)

pickle.dump((averaged, labels, d["seq_id"].tolist()), open("/data/xhorvat9/ltr_bert/NewClassifiers/Lineage/BERT/LTRBERT_superfamily_classifier_embeddings.b_average", "wb"))

