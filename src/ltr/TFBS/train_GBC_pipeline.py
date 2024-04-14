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
from utils.TFBS_utils import get_presence_count_dict

LTR_motifs = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/training_LTR_TFBS_old638.b", "rb"))
non_LTR_motifs = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/training_non_LTR_TFBS_long.b", "rb"))

# Load the motifs for LTR and non-LTR sequences
LTR_motif_dict_count = dict([(key, []) for key in LTR_motifs[list(LTR_motifs.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in LTR_motifs[list(LTR_motifs.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, LTR_motifs)


IDs = list(LTR_motifs.keys())
LTR_dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)

non_LTR_motif_dict_count = dict([(key, []) for key in non_LTR_motifs[list(non_LTR_motifs.keys())[0]]])
non_LTR_motif_dict_presence = dict([(key, []) for key in non_LTR_motifs[list(non_LTR_motifs.keys())[0]]])
get_presence_count_dict(non_LTR_motif_dict_count, non_LTR_motif_dict_presence, non_LTR_motifs)

IDs = list(non_LTR_motifs.keys())
non_LTR_dt = pd.DataFrame(non_LTR_motif_dict_count, index=IDs)

labels = [1] * len(LTR_dt) + [0] * len(non_LTR_dt)
data = pd.concat([LTR_dt, non_LTR_dt], axis=0)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)


# Create the training pipelines
tfidf_pipeline = Pipeline([("transformer",  TfidfTransformer()),
                           ("classifier", GradientBoostingClassifier(max_depth=8, min_samples_leaf=50, n_estimators=400))])

# Execute gridsearch
grid = tfidf_pipeline.fit(X_train, y_train)

# Print the best parameters and score
print("F1 Score ", f1_score(y_test, tfidf_pipeline.predict(X_test)))

import pickle
pickle.dump(tfidf_pipeline, open("GBC_pipeline.b", "wb+"))
