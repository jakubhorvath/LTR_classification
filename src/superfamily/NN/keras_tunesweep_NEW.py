import keras
import keras_tuner as kt
import tensorflow as tf
import Bio.SeqIO as SeqIO
import random
import numpy as np
import sys
import pandas as pd
import tqdm
from keras.models import Sequential 
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import wandb
from wandb.keras import WandbCallback
from utils.CNN_utils import remove_N, onehote

max_len=600
sequences  = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta") if len(rec.seq) < max_len and len(rec.seq) > 0 and rec.description.split()[3] != "NAN"]

labels = [rec.description.split()[3] for rec in sequences]
le = LabelEncoder()
labels = list(le.fit_transform(labels))
sequences = [onehote(remove_N(str(rec.seq))) for rec in sequences]


# Split into train and test
paddedDNA = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="pre", maxlen=max_len)
trainX, valX, trainY, valY = train_test_split(paddedDNA, labels, test_size=0.15, random_state=42)

label_weights = class_weight.compute_class_weight( class_weight='balanced', classes=np.unique(labels), y=labels)
weights = {c:w for c, w in zip(np.unique(labels), label_weights)}


######### Configure Sweep #########
sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'binary_crossentropy',
      'goal': 'minimize'   
    },
    'parameters': {
        'filters': {
            'values': [32, 64, 128]
        },
        'kernel_size': {
            'values': [8, 16, 32]
        },
        'dropout': {
            'values': [0.2, 0.3, 0.4]
        },
        'pool_size': {
            'values': [2, 4, 8]
        },
        'lstm_units': {
            'values': [50, 100, 150]
        },
        'optimizer': {
            'values': ['adam', 'nadam', 'rmsprop']
        }
    }
}
sweep_id = wandb.sweep(sweep_config, entity="diplomovka", project="Superfamily_KT_sweep_short")
######### Begin Training #########
def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'filters': 32,
        'kernel_size': 8,
        'dropout': 0.2,
        'pool_size': 2,
        'lstm_units': 50,
        'optimizer': 'adam'
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    model2 = Sequential()

    model2.add(Conv1D(filters=config.filters, kernel_size=config.kernel_size, padding='same', activation='relu', input_shape=trainX[0].shape))
    model2.add(Dropout(config.dropout))  # You can adjust the dropout rate as needed
    model2.add(MaxPooling1D(pool_size=config.pool_size))
    model2.add(LSTM(config.lstm_units))
    model2.add(Dense(units=1, activation='sigmoid'))

    model2.compile(loss='binary_crossentropy', optimizer=config.optimizer, metrics=['binary_accuracy'], weighted_metrics=["binary_accuracy"])

    #model2.fit(valX, np.array(valY), epochs=3, batch_size=64,verbose = 1,validation_data=(valX, np.array(valY)), callbacks=[WandbCallback()])
    model2.fit(trainX, np.array(trainY), epochs=15, batch_size=64,verbose = 1,validation_data=(valX, np.array(valY)), callbacks=[EarlyStopping(monitor='val_loss', patience=3), WandbCallback(validation_data=(valX, valY))], class_weight=weights)

wandb.agent(sweep_id, train)