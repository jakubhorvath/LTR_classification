import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
import pickle
import Bio.SeqIO as SeqIO
from utils.BERT_utils import *
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help='Path to model')
parser.add_argument('--num_labels', default=2, help='Number of labels in the model')
parser.add_argument('--in_seq_file', help='Path to input sequence file')
parser.add_argument('--out_path', help='Path to output file')
args = parser.parse_args()

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model.to(device)

min_len = 512
MAX_LEN = 4000
long_sequences = [str(rec.seq) for rec in SeqIO.parse(f"{args.in_seq_file}","fasta") if len(rec.seq) >= min_len and len(rec.seq) <= MAX_LEN]

n_sequences = len(long_sequences)

seq_ids = [rec.id for rec in SeqIO.parse(f"{args.in_seq_file}","fasta") if len(rec.seq) >= min_len]



# Split the sequences into windows
window_size = min_len
stride = min_len//3 # ~ 1/3 of window size

outputs = []
sequences = []

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
model.eval()
for s in tqdm.tqdm(sequences):
  # Tokenize the sequences
  tokenized_segment = tokenizer([tok_func(sequence_segment) for sequence_segment in s], padding=True, truncation=True, max_length=min_len, return_tensors="pt")
  tokenized_segment.to(device)
  with torch.no_grad():
      # Run model on segments
      outputs = model(**tokenized_segment, output_hidden_states=True)
      hidden_states = outputs.hidden_states
      embeddings = hidden_states[layer_index][:, token_index, :]
      embeddings = embeddings.to("cpu")
      model_embeddings.append(embeddings)

averaged = []
for emb in tqdm.tqdm(model_embeddings):
    averaged.append((sum(emb)/len(emb)).numpy())
averaged = np.array(averaged)

pickle.dump((averaged, seq_ids), open(args.out_path, "wb"))