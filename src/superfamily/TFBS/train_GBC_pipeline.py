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

# Load the detected motifs
motifs = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/training_LTR_TFBS.b", "rb"))
#motifs = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/sequence_motifs_sampled.b", "rb"))

# Create a dictionary to store the motif count and presence
LTR_motif_dict_count = dict([(key, []) for key in motifs[list(motifs.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in motifs[list(motifs.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, motifs)

# Create a dataframe from the dictionary
IDs = list(motifs.keys())
dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)

# Load the superfamilies
records = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta")]
superfamilies = [rec.description.split()[3] for rec in records]
IDs = [rec.id for rec in records]
superfam_df = pd.DataFrame({ "superfamily": superfamilies}, index=IDs)
superfam_df = superfam_df[superfam_df["superfamily"] != "NAN"]

data = dt.join(superfam_df, how="inner")

# Split the data
labels = data["superfamily"]
le = LabelEncoder()
labels = list(le.fit_transform(labels))
X_train, X_test, y_train, y_test = train_test_split(data.drop("superfamily", axis=1), labels, test_size=0.2, random_state=42)


# Create the training pipelines
tfidf_pipeline = Pipeline([("transformer",  TfidfTransformer()),
                           ("classifier", GradientBoostingClassifier(max_depth=8, min_samples_leaf=50, n_estimators=400))])

# Execute gridsearch
grid = tfidf_pipeline.fit(X_train, y_train)

# Print the best parameters and score
print("F1 Score ", f1_score(y_test, tfidf_pipeline.predict(X_test)))

import pickle
pickle.dump(tfidf_pipeline, open("GBC_pipeline.b", "wb+"))
