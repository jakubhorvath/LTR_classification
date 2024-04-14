import pickle
from sklearn.metrics import f1_score
mot = pickle.load(open("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/training_LTR_TFBS.b", "rb"))

def get_presence_count_dict(motif_dict_count, motif_dict_presence, TF_sites):
    for seq in TF_sites:
        for motif in motif_dict_count:
            if len(TF_sites[seq][motif]) > 0:
                motif_dict_count[motif].append(len(TF_sites[seq][motif]))
                motif_dict_presence[motif].append(1)
            else:
                motif_dict_count[motif].append(0)
                motif_dict_presence[motif].append(0)


LTR_motif_dict_count = dict([(key, []) for key in mot[list(mot.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in mot[list(mot.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, mot)

import pandas as pd
import Bio.SeqIO as SeqIO
IDs = list(mot.keys())
dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)
records = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta")]
rec_lineages = [rec.description.split()[4] for rec in records]
IDs = [rec.id for rec in records]
lineage_df = pd.DataFrame({ "label": rec_lineages}, index=IDs)
lineage_df = lineage_df[lineage_df["label"] != "NAN"]
lineage_df = lineage_df[lineage_df["label"].isin(list(lineage_df["label"].value_counts().iloc[:13].index))]
data = pd.DataFrame({'sequence':[str(rec.seq) for rec in records], 'label':[rec.description.split(" ")[4] for rec in records], "seq_id":[rec.id for rec in records]})

data = dt.join(lineage_df, how="inner")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
labels = data["label"]
le = LabelEncoder()
labels = list(le.fit_transform(labels))

X_train, X_test, y_train, y_test = train_test_split(data.drop("label", axis=1), labels, test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
# Create the training pipelines
tfidf_pipeline = Pipeline([("transformer",  TfidfTransformer()),
    ("classifier", GradientBoostingClassifier(max_depth=8, min_samples_leaf=50, n_estimators=400, verbose=1))])

# Execute gridsearch
grid = tfidf_pipeline.fit(X_train, y_train)

# Print the best parameters and score
print("F1 Score ", f1_score(y_test, tfidf_pipeline.predict(X_test), average='weighted'))

import pickle
pickle.dump(tfidf_pipeline, open("GBC_pipeline.b", "wb+"))
pickle.dump(le, open("label_encoder.b", "wb+"))