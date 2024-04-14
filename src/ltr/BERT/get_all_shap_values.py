import torch
import shap
import transformers
import torch
import numpy as np
import scipy as sp
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import Bio.SeqIO as SeqIO
import pickle
import random

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

 # the kmer splitting function
def Kmers_funct(seq, size=6):
   return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]
def tok_func(x): return " ".join(Kmers_funct(x))


# load a BERT sentiment analysis model
tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
model = BertForSequenceClassification.from_pretrained("/data/xhorvat9/ltr_bert/NewClassifiers/LTR_classifier/BERT/LTRBERT_LTR_classifier_512")
model = model.to(device)

def f(x):
    """
    This function is passed to the SHAP explainer
    It preprocesses the data and draws predictions using the loaded model
    """
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).to("cuda")
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

MAX_LEN=512
random.seed(10)
records  = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
LTR_sequences = [str(rec.seq) for rec in records]

n_sequences = len(LTR_sequences)

generated, genomic, markov = int(n_sequences*0.15), int(n_sequences*0.6), int(n_sequences*0.25)

genomic_non_LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/non_LTRs_training_genomic_extracts.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
if genomic < len(genomic_non_LTRs):
    genomic_non_LTRs = random.sample(genomic_non_LTRs, genomic)
generated_non_LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/non_LTRs_training_generated.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
if generated < len(generated_non_LTRs):
    generated_non_LTRs = random.sample(generated_non_LTRs, generated)
markov_non_LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/non_LTRs_training_markovChain.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
if markov < len(markov_non_LTRs):
    markov_non_LTRs = random.sample(markov_non_LTRs, markov)
non_LTRs = genomic_non_LTRs + generated_non_LTRs + markov_non_LTRs
non_LTRs = [str(rec.seq) for rec in non_LTRs]


non_LTRs = random.sample(non_LTRs, 5000)
LTR_sequences = random.sample(LTR_sequences, 5000)

records = LTR_sequences + non_LTRs

# Runs the SHAP explainer
explainer = shap.Explainer(f, tokenizer)
shap_values = explainer([tok_func(v) for v in records])

pickle.dump(shap_values, open("./shap_values_sampled.b", "wb+"))