import tensorflow as tf
from sklearn.model_selection import train_test_split
import Bio.SeqIO as SeqIO
import random
import tqdm
import numpy as np
import shap
from shap import DeepExplainer
import pickle
#tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(False)
import os
model = tf.keras.models.load_model("/data/xhorvat9/ltr_bert/NewClassifiers/Lineage/NN/all_length_cnn_lstm.h5")
#short_model = tf.keras.models.load_model('/data/xhorvat9/ltr_bert/NewClassifiers/LTR_classifier/NN/short_seq_cnn_lstm.h5')



def remove_N(seq):
    """
    Remove Ns from sequence
    """
    return seq.upper().replace("N", "")

def onehote(seq):
    """
    One Hot encoding function
    """
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0, 0., 1., 0.], "T":[0., 0., 0., 1.], "N":[0., 0., 0., 0.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
    return np.array(seq2)

import numpy as np
MAX_LEN=4000
MIN_LEN=0
random.seed(10)
LTRs = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/test_LTRs.fasta", "fasta") if len(rec.seq) < MAX_LEN and len(rec.seq) > MIN_LEN]
LTRs = random.sample(LTRs, 20000)


sequences = [onehote(remove_N(str(rec.seq))) for rec in tqdm.tqdm(LTRs)]


LTRs = random.sample(LTRs, 200)
train_seqs = [onehote(remove_N(str(rec.seq))) for rec in tqdm.tqdm(LTRs)]

training_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding="pre", maxlen=3000)
# Split into train and test
paddedDNA = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="pre", maxlen=3000)


#tf.compat.v1.disable_v2_behavior()
np.random.seed(1)
#paddedDNA = paddedDNA[-len(LTRs):]

n=10

print("Creating DeepExplainer object")
seq_explainer = shap.DeepExplainer(model, training_padded)
mean = np.zeros((n,3000))
print("Calculating SHAP values")
shap_values = seq_explainer.shap_values(paddedDNA)
#for i in tqdm.tqdm(range(0, len(paddedDNA), n)):
#    seqs_to_explain = paddedDNA[i:i+n]# these three are positive for task 0
#    shap_values = seq_explainer.shap_values(seqs_to_explain)
#    mean += np.mean(shap_values[0], axis=0)
#    print("Done with", i)
pickle.dump(shap_values, open("shap_values_40000.b", "wb+"))

#mean = mean/(len(paddedDNA)/1000)
