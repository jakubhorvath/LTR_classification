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
model = BertForSequenceClassification.from_pretrained("/data/xhorvat9/ltr_bert/NewClassifiers/Superfamily/BERT/LTRBERT_superfamily_512")
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
records  = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > 0]
LTR_sequences = [str(rec.seq) for rec in records]

records = LTR_sequences #+ non_LTRs

# Runs the SHAP explainer
explainer = shap.Explainer(f, tokenizer)
shap_values = explainer([tok_func(v) for v in records])

pickle.dump(shap_values, open("./shap_values_all.b", "wb+"))