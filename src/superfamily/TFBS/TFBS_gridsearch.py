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


LTR_dt = pd.read_csv("/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/LTR_motif_counts.csv", index_col=0)


records = [rec for rec in SeqIO.parse("/var/tmp/xhorvat9/ltr_bert/FASTA_files/train_LTRs.fasta", "fasta")]
superfamilies = [rec.description.split()[3] for rec in records]
IDs = [rec.id for rec in records]
superfam_df = pd.DataFrame({ "superfamily": superfamilies}, index=IDs)
superfam_df = superfam_df[superfam_df["superfamily"] != "NAN"]

data = LTR_dt.join(superfam_df, how="inner")


labels = data["superfamily"]
le = LabelEncoder()
labels = list(le.fit_transform(labels))

X_train, X_test, y_train, y_test = train_test_split(data.drop("superfamily", axis=1), labels, test_size=0.2, random_state=42)


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
grid = GridSearchCV(tfidf_pipeline, parameters, verbose = 10, n_jobs=1,scoring ="balanced_accuracy").fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters: ", grid.best_params_)
print("Best Score: ", grid.best_score_)
print("F1 Score ", f1_score(y_test, grid.predict(X_test)))
