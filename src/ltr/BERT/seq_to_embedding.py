import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
import pickle

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained("./LTRBERT_LTR_classifier_512", num_labels=2)

if torch.cuda.is_available():    

    device = torch.device("cuda")

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model.to(device)

# write helper functions for tokenizing data
 # the kmer splitting function
def Kmers_funct(seq, size=6):
   return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]
def tok_func(x): return " ".join(Kmers_funct(x.replace("N","")))


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
import Bio.SeqIO as SeqIO
from sklearn.model_selection import train_test_split

min_len = 512
MAX_LEN = 4000
sequences = [str(rec.seq) for rec in SeqIO.parse(f"/data/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta","fasta") if len(rec.seq) >= min_len and len(rec.seq) <= MAX_LEN]
#sequences = [str(rec.seq) for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta","fasta")]

n_sequences = len(sequences)

generated, genomic, markov = int(n_sequences*0.15), int(n_sequences*0.6), int(n_sequences*0.25)
random.seed(42)
genomic_non_LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/non_LTRs_test_genomic_extracts.fasta", "fasta") if len(rec.seq) >= min_len and len(rec.seq) <= MAX_LEN]
if genomic < len(genomic_non_LTRs):
    genomic_non_LTRs = random.sample(genomic_non_LTRs, genomic)
generated_non_LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/non_LTRs_test_generated.fasta", "fasta") if len(rec.seq) >= min_len and len(rec.seq) <= MAX_LEN]
if generated < len(generated_non_LTRs):
    generated_non_LTRs = random.sample(generated_non_LTRs, generated)
markov_non_LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/non_LTRs_test_markovChain.fasta", "fasta") if len(rec.seq) >= min_len and len(rec.seq) <= MAX_LEN]
if markov < len(markov_non_LTRs):
    markov_non_LTRs = random.sample(markov_non_LTRs, markov)
non_LTRs = genomic_non_LTRs + generated_non_LTRs + markov_non_LTRs

seq_ids = [rec.id for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta","fasta") if len(rec.seq) >= min_len] + [rec.id for rec in non_LTRs]
non_LTRs = [str(rec.seq) for rec in non_LTRs]

long_sequences = sequences + non_LTRs
labels = np.array([1] * len(sequences) + [0]* len(non_LTRs))

#X_train, X_test, y_train, y_test = train_test_split(long_sequences, labels, random_state=42, test_size=0.3)

from transformers import Trainer
import random
import tqdm
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
  
# Get the embeddings for the last layer and a specific token (e.g., the first token)
layer_index = -1  # Index of the last layer
token_index = 0   # Index of the token you're interested in
model_embeddings = []
model.eval()
for s in tqdm.tqdm(sequences):
  tokenized_segment = tokenizer([tok_func(sequence_segment) for sequence_segment in s], padding=True, truncation=True, max_length=min_len, return_tensors="pt")
  tokenized_segment.to(device)
  with torch.no_grad():
      outputs = model(**tokenized_segment, output_hidden_states=True)
      hidden_states = outputs.hidden_states
      embeddings = hidden_states[layer_index][:, token_index, :]
      embeddings = embeddings.to("cpu")
      model_embeddings.append(embeddings)
import pickle

import tqdm
averaged = []
for emb in tqdm.tqdm(model_embeddings):
    averaged.append((sum(emb)/len(emb)).numpy())
averaged = np.array(averaged)

pickle.dump((averaged, labels, seq_ids), open("/data/xhorvat9/ltr_bert/NewClassifiers/LTR_classifier/BERT/LTRBERT_LTR_classifier_embeddings_TEST.b_average", "wb"))