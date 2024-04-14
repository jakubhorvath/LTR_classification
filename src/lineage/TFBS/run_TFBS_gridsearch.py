import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import random
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import Bio.SeqIO as SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def get_presence_count_dict(motif_dict_count, motif_dict_presence, TF_sites):
    for seq in TF_sites:
        for motif in motif_dict_count:
            if len(TF_sites[seq][motif]) > 0:
                motif_dict_count[motif].append(len(TF_sites[seq][motif]))
                motif_dict_presence[motif].append(1)
            else:
                motif_dict_count[motif].append(0)
                motif_dict_presence[motif].append(0)

n_classes = 15
motifs = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/training_LTR_TFBS_old638.b", "rb"))

#motifs = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/sequence_motifs_sampled.b", "rb"))
#LTR_dt = pd.read_csv("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/LTR_motif_counts.csv", index_col=0)
LTR_motif_dict_count = dict([(key, []) for key in motifs[list(motifs.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in motifs[list(motifs.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, motifs)

import pandas as pd
IDs = list(motifs.keys())
dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)

import Bio.SeqIO as SeqIO
records = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta")]
rec_ids = [rec.id for rec in records]
rec_lineages = [rec.description.split()[4] for rec in records]

lineages = [rec.description.split()[4] for rec in records]
IDs = [rec.id for rec in records]
lineage_df = pd.DataFrame({ "lineage": lineages}, index=IDs)
d = dt.join(lineage_df, how="inner")

d = d[~d['lineage'].str.contains("copia")]
d = d[d["lineage"].isin(d["lineage"].value_counts()[:n_classes].index.tolist())]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(d['lineage'])
pickle.dump(label_encoder, open("/var/tmp/xhorvat9/ltr_bert/NewClassifiers/Lineage/TFBS/label_encoder.b", "wb+"))

trainX, valX, trainY, valY = train_test_split(d.drop("lineage", axis=1), labels, test_size=0.2, random_state=42)

# Create the training pipelines
tfidf_pipeline = Pipeline([("transformer",  TfidfTransformer()),
                           ("classifier", RandomForestClassifier())])

# Set the parameters for the gridsearch
parameters = [
{
	'classifier': (MLPClassifier(solver='adam', activation='logistic', early_stopping=True, validation_fraction=0.1),),
    'classifier__learning_rate_init' : [0.1, 0.05, 0.02, 0.01],
    'classifier__hidden_layer_sizes' : [(10,), (50,), (100,), (200,)],
    'classifier__alpha': [0.0001, 0.001, 0.01],
},
{
	'classifier': (GradientBoostingClassifier(),),
    'classifier__n_estimators' : [50, 100, 200, 400],
    'classifier__learning_rate' : [0.1, 0.05, 0.02, 0.01],
    'classifier__max_depth' : [4, 6, 8],
    'classifier__min_samples_leaf' : [20, 50,100,150]
},
{
	'classifier': (RandomForestClassifier(),),
    'classifier__n_estimators' : [100, 300, 600],
    'classifier__max_depth' : [4, 6, 8, 10, 12],
}]
# Execute gridsearch
grid = GridSearchCV(tfidf_pipeline, parameters, verbose = 10, n_jobs=1,scoring ="balanced_accuracy").fit(trainX, trainY)

# Print the best parameters and score
print("Best Parameters: ", grid.best_params_)
print("Best Score: ", grid.best_score_)
print("F1 Score ", f1_score(valY, grid.predict(valX)))
