import tensorflow as tf
from sklearn.model_selection import train_test_split
import Bio.SeqIO as SeqIO
import random
import tqdm
import numpy as np
import shap
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to model')
parser.add_argument('--LTR_seq_file', help='Path to LTR fasta file')
parser.add_argument('--non_LTR_seq_file', help='Path to nonLTR fasta file')
args = parser.parse_args()

tf.compat.v1.disable_eager_execution()
import os
model = tf.keras.models.load_model(args.model)
from utils.CNN_utils import remove_N, onehote
import numpy as np

MAX_LEN=4000
MIN_LEN=0
random.seed(10)
LTRs = [rec for rec in SeqIO.parse(args.LTR_seq_file, "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > MIN_LEN]
non_LTRs = [rec for rec in SeqIO.parse(args.non_LTR_seq_file, "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > MIN_LEN]
LTRs = random.sample(LTRs, 20000)
non_LTRs = random.sample(non_LTRs, 20000)


sequences = [onehote(remove_N(str(rec.seq))) for rec in tqdm.tqdm(LTRs+non_LTRs)]
LTRs = random.sample(LTRs, 3000)
non_LTRs = random.sample(non_LTRs, 3000)
train_seqs = [onehote(remove_N(str(rec.seq))) for rec in tqdm.tqdm(LTRs+non_LTRs)]

training_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding="pre", maxlen=3000)
# Split into train and test
paddedDNA = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="pre", maxlen=3000)

np.random.seed(1)
n=10

print("Creating DeepExplainer object")
seq_explainer = shap.DeepExplainer(model, training_padded)
mean = np.zeros((n,3000))
print("Calculating SHAP values")
shap_values = seq_explainer.shap_values(paddedDNA)
pickle.dump(shap_values, open("shap_values_40000.b", "wb+"))
