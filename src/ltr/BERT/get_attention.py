import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import Bio.SeqIO as SeqIO 
import random
import seaborn as sns
import numpy as np
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
device = "cpu"


# load a BERT sentiment analysis model
tokenizer = BertTokenizer.from_pretrained(f'zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained("/data/xhorvat9/ltr_bert/NewClassifiers/LTR_classifier/BERT/LTRBERT_LTR_classifier_512", num_labels=2, output_hidden_states=True, output_attentions=True)
model = model.to(device)

 # the kmer splitting function
def Kmers_funct(seq, size=6):
   return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]
def tok_func(x): return " ".join(Kmers_funct(x))

max_len = 512
random.seed(42)
n_sequences = 1000
records  = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta", "fasta") if len(rec.seq) < max_len and len(rec.seq) > 450]
LTR_sequences = random.sample([str(rec.seq) for rec in records],n_sequences)
seqs = [tok_func(v) for v in LTR_sequences]
tv = [tokenizer(s, padding='max_length', max_length=512, truncation=True, return_tensors='pt') for s in seqs]

# A bit complicated to handle cuda memory errors
batch_process_size = 50
index = 0
model.eval()
while index < len(tv):
    with torch.no_grad():
        process = tv[index].to(device)
        output = model(**process)
    if index == 0:
        attentions = torch.stack(output['attentions'], dim=0).to("cpu")
    else:
        attentions = torch.cat((attentions, torch.stack(output['attentions'], dim=0).to("cpu")), dim=1)
    index += 1
    print(index)
attentions.shape #(layer, batch_size (squeezed by torch.cat), num_heads, sequence_length, sequence_length)

import pickle
#pickle.dump(attentions, open("LTR_BERT_attentions.pkl", "wb"))
# Normalize the attentions and output to file
first_layer = attentions[0, :, :, :, :]
kmer=6
scores = np.zeros([first_layer.shape[0], first_layer.shape[-1]])

for index, attention_score in enumerate(first_layer):
    attn_score = []
    for i in range(1, attention_score.shape[-1] - kmer + 2):
        attn_score.append(float(attention_score[:, 0, i].sum()))

    for i in range(len(attn_score) - 1):
        if attn_score[i + 1] == 0:
            attn_score[i] = 0
            break

    counts = np.zeros([len(attn_score) + kmer - 1])
    real_scores = np.zeros([len(attn_score) + kmer - 1])
    for i, score in enumerate(attn_score):
        for j in range(kmer):
            counts[i + j] += 1.0
            real_scores[i + j] += score
    real_scores = real_scores / counts
    real_scores = real_scores / np.linalg.norm(real_scores)

    scores[index] = real_scores
print(scores.shape)
np.save("./LTR_BERT_attentions.npy", scores)
sns.heatmap(scores)
