import pickle 
import pandas as pd
import Bio.SeqIO as SeqIO
import shap
from utils.TFBS_utils import get_presence_count_dict

# Load the trained GBC pipeline and extract vectorizer and classifier
pipeline = pickle.load(open('/data/xhorvat9/ltr_bert/NewClassifiers/Lineage/TFBS/GBC_pipeline.b', 'rb'))
TFIDF_transformer = pipeline["transformer"]
GBC = pipeline["classifier"]
test_features = pickle.load(open("/data/xhorvat9/ltr_bert/Simple_ML_model/test_LTR_TFBS_old638.b", "rb"))

# Load features into a dictionary
LTR_motif_dict_count = dict([(key, []) for key in test_features[list(test_features.keys())[0]]])
LTR_motif_dict_presence = dict([(key, []) for key in test_features[list(test_features.keys())[0]]])
get_presence_count_dict(LTR_motif_dict_count, LTR_motif_dict_presence, test_features)

n_classes = 15
IDs = list(test_features.keys())
dt = pd.DataFrame(LTR_motif_dict_count, index=IDs)

# Load the corresponding lineages
records = [rec for rec in SeqIO.parse("/data/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta")]
rec_ids = [rec.id for rec in records]


lineages = [rec.description.split()[4] for rec in records]
IDs = [rec.id for rec in records]
lineage_df = pd.DataFrame({ "lineage": lineages}, index=IDs)
data = dt.join(lineage_df, how="inner")
data = data[~data['lineage'].str.contains("copia")]
data = data[data["lineage"].isin(data["lineage"].value_counts()[:n_classes].index.tolist())]


# Load the label encoder and transform the labels
labels = data["lineage"]
label_encoder = pickle.load(open("/data/xhorvat9/ltr_bert/NewClassifiers/Lineage/label_encoder.b", "rb"))
labels = label_encoder.transform(data['lineage'])
data = data.iloc[:, :-1]

# Run the explainer and save the shap values
explainer = shap.TreeExplainer(GBC)
shap_values = explainer.shap_values(data)

pickle.dump(shap_values, open("shap_values.b", "wb+"))