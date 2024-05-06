import pickle
from sklearn.metrics import f1_score
from utils.TFBS_utils import get_presence_count_dict
import argparse
import pandas as pd
import Bio.SeqIO as SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the model')
parser.add_argument('--LTR_motif_file', help='Path to LTR motif file')
parser.add_argument('--LTR_fasta_file', help='Path to LTR fasta file')
args = parser.parse_args()

mot = pickle.load(open(args.LTR_motif_file, "rb"))

# Create a dictionary to store the motif count and presence
LTR_motif_dict_count = dict([(key, []) for key in mot[list(mot.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in mot[list(mot.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, mot)

# Assign lineages to the LTRs
IDs = list(mot.keys())
dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)
records = [rec for rec in SeqIO.parse(args.LTR_fasta_file, "fasta")]
rec_lineages = [rec.description.split()[4] for rec in records]
IDs = [rec.id for rec in records]
lineage_df = pd.DataFrame({ "label": rec_lineages}, index=IDs)
lineage_df = lineage_df[lineage_df["label"] != "NAN"]
lineage_df = lineage_df[lineage_df["label"].isin(list(lineage_df["label"].value_counts().iloc[:13].index))]
data = pd.DataFrame({'sequence':[str(rec.seq) for rec in records], 'label':[rec.description.split(" ")[4] for rec in records], "seq_id":[rec.id for rec in records]})

data = dt.join(lineage_df, how="inner")

# Split the data and encode labels
labels = data["label"]
le = LabelEncoder()
labels = list(le.fit_transform(labels))

X_train, X_test, y_train, y_test = train_test_split(data.drop("label", axis=1), labels, test_size=0.2, random_state=42)

# Create the training pipelines
tfidf_pipeline = Pipeline([("transformer",  TfidfTransformer()),
    ("classifier", GradientBoostingClassifier(max_depth=8, min_samples_leaf=50, n_estimators=400, verbose=1))])

# Execute gridsearch
grid = tfidf_pipeline.fit(X_train, y_train)

# Print the best parameters and score
print("F1 Score ", f1_score(y_test, tfidf_pipeline.predict(X_test), average='weighted'))

pickle.dump(tfidf_pipeline, open("GBC_pipeline.b", "wb+"))
pickle.dump(le, open("label_encoder.b", "wb+"))