import torch
import shap
import transformers
import torch
import random 
from transformers import BertTokenizer, BertForSequenceClassification
from utils import BERT_utils
import Bio.SeqIO as SeqIO 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the model')
parser.add_argument('--LTR_seq_file', help='Path to LTR fasta file')
args = parser.parse_args()

if torch.cuda.is_available():     
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
device = "cpu"

LTR_sequences = []

# Remove TSDs to prevent SHAP attributing high values and ignore other parts of sequence
LTR_sequences = [str(rec.seq)[3:-3] for rec in SeqIO.parse(args.LTR_seq_file, "fasta")]
LTR_sequences = [s for s in LTR_sequences if len(s) < 512]

random.seed(42)
LTR_sequences = random.sample(LTR_sequences, 3000)


tokenizer = BertTokenizer.from_pretrained('zhihan1996/DNA_bert_6')
m = BertForSequenceClassification.from_pretrained(args.model, num_labels=15)

m = m.to(device)

# Model the shap pipeline as sentiment analysis (multi-class classification)
pred = transformers.pipeline('sentiment-analysis',
 model=m,
  tokenizer=tokenizer,
  return_all_scores=True,
  device=device)

explainer = shap.Explainer(pred)

shap_values = explainer([BERT_utils.tok_func(x) for x in LTR_sequences])

import pickle 
pickle.dump(shap_values, open("./shap_values_lineage_3000seqs.b", "wb"))
