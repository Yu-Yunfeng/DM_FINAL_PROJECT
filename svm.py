import dt
import numpy as np
from sklearn.preprocessing import label_binarize;
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier;
from sklearn import svm
from sklearn.metrics import precision_recall_curve, average_precision_score,recall_score,precision_score,accuracy_score

random_state = np.random.RandomState(0);

data,dic = dt.trans_to_num();
print(data);
y = data["readmitted"]
X = data.drop(labels=["readmitted"],axis=1)

Y = label_binarize(y,classes=[0,1,2]);
n_classes = Y.shape[1];

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4);
classifier = OneVsRestClassifier(svm.LinearSVC(penalty='l1',dual=False));
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
    print(recall_score(test_l,y_score,average='weighted'));
    print(precision_score(test_l,y_score,average='micro'))
    print(precision_score(test_l,y_score,average='micro'))


#print(train_set);
#print(test_set);
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=random_state);
# train_l = train_set['readmitted'];
# test_l = test_set['readmitted'];
# train_f = train_set.drop(labels = ['readmitted'],axis=1);
# test_f = test_set.drop(labels = ['readmitted'],axis=1);



# classifier.fit(train_f,train_l);
# y_score = classifier.predict(test_f);
#
# y_score = label_binarize(y_score,classes=[0,1,2])
# test_l = label_binarize(test_l,classes=[0,1,2])

# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(test_l[:, i],
#                                                         y_score[:, i])
#     average_precision[i] = average_precision_score(test_l[:, i], y_score[:, i])
#
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(test_l.ravel(),
#     y_score.ravel())
# average_precision["micro"] = average_precision_score(test_l, y_score,
#                                                      average="micro")

# print(precision);
# print(recall);
# print(average_precision)