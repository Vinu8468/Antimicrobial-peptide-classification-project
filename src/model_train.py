import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import itertools 
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# feature is aa count

def compute_aac(sequence):
    length = len(sequence)
    freq = {}
    for aa in AMINO_ACIDS:
        freq[aa] = sequence.count(aa)/length
    return freq

# adding charge component
def compute_net_charge(sequence):
    charge_dict = {
        "K":1,"R":1,"H":0.1,"D":-1,"E":-1 # negative and positive animo acids get a score
    }
    charge=0
    for aa in sequence:
        charge += charge_dict.get(aa,0)
    return charge/len(sequence) # this is just to normalize

# hydrophobicity .. since it has an influence on membranes 
# below is kyte-doolittle scale
hydrophobicity_dict= {
    "A":1.8,"C":2.5,"D":-3.5,"E":-3.5,
    'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M': 1.9, 'N':-3.5,
    'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
    'T':-0.7, 'V':4.2, 'W':-0.9,'Y':-1.3
}

def compute_hydrophobicity(sequence):
    return sum(hydrophobicity_dict.get(aa,0)for aa in sequence)/len(sequence)


# now the dipeptide can also be considered
import itertools
dipeptides = [''.join(dp) for dp in itertools.product(AMINO_ACIDS,repeat=2)]

def compute_dipeptide(sequence):
    length = len(sequence)
    counts = dict.fromkeys(dipeptides,0)
    
    for i in range(length -1):
        dp = sequence[i:i+2]
        if dp in counts:
            counts[dp]+=1

    #normalize
    for dp in counts:
        counts[dp]/=(length-1)
    return counts

# tripeptides would help but .. 20^3 = 8000 features thats too much

# load data
df = pd.read_csv("../data/processed/final_dataset_v2.csv")

print("dataset loaded:",df.shape)


def extract_features(sequence):
    features = {}
    # AAC 
    features.update(compute_aac(sequence))
    
    #net charge
    features["net_charge"]= compute_net_charge(sequence)

    # hydrophobicity
    features["hydrophobicity"] = compute_hydrophobicity(sequence)

    # dipeptide
    features.update(compute_dipeptide(sequence))

    return features

#feature extraction
features = df["sequence"].apply(extract_features)
features_df = pd.DataFrame(features.tolist())

X = features_df
y = df["label"]

print("Feature shape:",X.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state = 42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train size ", X_train.shape)
print("Test size ", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter = 1000,C=1,penalty="l2"),
    'Support Vector Classifier (rbf kernal)': SVC(kernel="rbf",C=10,gamma="scale",probability= True),
    "Random Forest Classifier":RandomForestClassifier(n_estimators = 200,max_depth = None,min_samples_split=2,random_state =42),
    "XG boost classifier": XGBClassifier(n_estimators= 200,learning_rate=0.1,max_depth=5,subsample=0.8,colsample_bytree= 0.8,use_label_encoder=False,eval_metric ="logloss")
}

for name, model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} results:")
    print(classification_report(y_test,y_pred))
    print(f"accuracy : {accuracy_score(y_test,y_pred)}")
    print(f"confussion matrix: \n{confusion_matrix(y_test,y_pred)}")
    print(f"f1 score: {f1_score(y_test,y_pred)}")
