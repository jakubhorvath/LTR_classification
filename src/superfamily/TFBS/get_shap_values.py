import pickle 
import pandas as pd
import Bio.SeqIO as SeqIO
from sklearn.preprocessing import LabelEncoder
import shap 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--pipeline_path', help='Path to the model')
parser.add_argument('--input_features', help='Path to input features file')
parser.add_argument('--seq_file', help='Path to sequence file')
parser.add_argument('--out_path', help='Path to output file')
args = parser.parse_args()

pipeline = pickle.load(open(args.pipeline_path, 'rb'))
TFIDF_transformer = pipeline["transformer"]
GBC = pipeline["classifier"]
test_features = pickle.load(open(args.input_features, "rb"))

def get_presence_count_dict(motif_dict_count, motif_dict_presence, TF_sites):
    for seq in TF_sites:
        for motif in motif_dict_count:
            if len(TF_sites[seq][motif]) > 0:
                motif_dict_count[motif].append(len(TF_sites[seq][motif]))
                motif_dict_presence[motif].append(1)
            else:
                motif_dict_count[motif].append(0)
                motif_dict_presence[motif].append(0)

# Load the TFBS occurences into a dictionary 
LTR_motif_dict_count = dict([(key, []) for key in test_features[list(test_features.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in test_features[list(test_features.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, test_features)


IDs = list(test_features.keys())
dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)

# Assign superfamily labels to the LTRs
records = [rec for rec in SeqIO.parse(args.seq_file, "fasta")]
superfamilies = [rec.description.split()[3] for rec in records]
IDs = [rec.id for rec in records]
superfam_df = pd.DataFrame({ "superfamily": superfamilies}, index=IDs)
superfam_df = superfam_df[superfam_df["superfamily"] != "NAN"]

data = dt.join(superfam_df, how="inner")


labels = data["superfamily"]
le = LabelEncoder()
labels = list(le.fit_transform(labels))
data = data.iloc[:, :-1]


explainer = shap.TreeExplainer(GBC)
shap_values = explainer.shap_values(data)

pickle.dump(shap_values, open(args.out_path, "wb+"))