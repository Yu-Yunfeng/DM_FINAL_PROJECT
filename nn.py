import dt;
import pandas as pd;
import sklearn.neural_network as nn;
from sklearn.model_selection import StratifiedShuffleSplit;
from sklearn.metrics import recall_score,precision_score,accuracy_score;
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
import numpy as np;


random_state = np.random.RandomState(0);

data,dic = dt.trans_to_num(one_hot_encoding=True);

if 'index' in data.keys():
    data.drop(labels=['index'],inplace=True,axis=1)

# print(data['readmitted'])
# corr = dt.get_correlation(data=data,att='readmitted');
# labels_irrelavant  = [];
# for key in corr.keys():
#     if abs(corr[key])<=0.01:
#         labels_irrelavant.append(key);
# data.drop(labels=labels_irrelavant,inplace=True,axis=1)

# print(corr)
y = data["readmitted"]
X = data.drop(labels=["readmitted"],axis=1)

print(data.info())

Y = label_binarize(y,classes=[0,1,2]);
n_classes = Y.shape[1];

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.4);
classifier = nn.MLPClassifier(hidden_layer_sizes=200)
#classifier = OneVsOneClassifier(svm.LinearSVC(penalty='l1',dual=False));
for train_index, test_index in sss.split(data, data['readmitted'],):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]
    train_l = train_set['readmitted'];
    test_l = test_set['readmitted'];
    train_f = train_set.drop(labels=['readmitted'], axis=1);
    test_f = test_set.drop(labels=['readmitted'], axis=1);
    classifier.fit(train_f, train_l);
    y_score = classifier.predict(test_f);
#    print(recall_score(test_l,y_score,average='weighted'));
    print(precision_score(test_l,y_score,average='weighted'))
    print(accuracy_score(test_l,y_score))

