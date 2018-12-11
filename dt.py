import pandas as pd;
import numpy as np;
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,average_precision_score

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit

#return data
def load_file(filename = 'diabetic_data.csv'):
    data = pd.read_csv(filename);
    data.replace({'?': 'NaN'},inplace=True);
    data.fillna(method="ffill",inplace=True)
#    print(data.isin(['?']))
#    print(data.info())
    return data;

#
# def deal_with_null(data = load_file(),threshold = 0.3):
#     att_to_be_deleted = ['weight', 'payer_code','examide','citoglipton','glimepiride-pioglitazone'];
#     data.drop(labels=att_to_be_deleted,axis=1,inplace=True);
#     data_array = np.array(data);
#     feature_num = len(data_array[0])-1;
#
#     feature_data = data_array[:,:feature_num];
#     label_data = data_array[:,feature_num:];
#     print(data_array)
#     return feature_data,label_data;

def trans_to_num(data = load_file(),dic = dict(),att_to_be_deleted = ['encounter_id','weight', 'payer_code','examide','citoglipton','glimepiride-pioglitazone'],one_hot_encoding = False):
    dic = dict();
    data.drop(labels=att_to_be_deleted,inplace=True,axis = 1);

    index = data.loc[data['discharge_disposition_id'] == (2 or 6)].index;
    data.drop(index=index,inplace=True);
    data.drop_duplicates(subset=['patient_nbr'],keep='first',inplace=True);
    data.drop(labels=['patient_nbr'],inplace=True,axis=1);

#    print(data.info())
    keys = data.keys();
    keys = keys[:-1]
    for key in keys:
        if data[key].dtype=='object':
            data[key],dic[key] = data[key].factorize();
            if(one_hot_encoding):
                if(data[key].nunique()>100):
                    continue;
                one_hot = pd.get_dummies(data[key],prefix=key);
                data.drop(key,axis=1,inplace=True);
                data = data.join(one_hot)
#                print(one_hot.keys())

    data.reset_index(inplace=True);
    return data,dic;

def generate_train_test_set(data = np.zeros((1,1)),train_ratio=0.7):
    data_size = len(data);
    train_size = int(data_size*train_ratio);
    return data[:train_size],data[train_size:];

def error(pre = [],real = []):
    size = len(pre);
    count = 0;
    for i in range(size):
        if(pre[i]==real[i]):
            count+=1;
    return count/size;

def get_correlation(data = load_file(),att = 'readmitted'):
    print(data.keys())
    corr_matrix = data.corr();
    res = corr_matrix[att].sort_values(ascending=True);
    return res

def avg_precision(y_test,y_pre):
    precision = dict();
    Y_test = label_binarize(y_test,classes=[0,1,2]);
    Y_pre = label_binarize(y_pre,classes=[0,1,2]);
    for i in range(3):
        precision[i] = average_precision_score(Y_test[:,i],Y_pre[:,i]);
    return precision;

def main():
#    print(data['readmitted'])
    data,dic = trans_to_num()
#    print(data.info())
    re_cor = get_correlation(data,att='readmitted')
    print(re_cor)
#     for key in data.keys():
#         if len(data[key].unique()) == 1:
#             print(key)
    # data = trans_to_num()[0]
    # print(data)
#    data.reset_index(inplace=True);
    labels = data['readmitted'];
    features = data.drop(labels=['readmitted',],axis=1);

    X = features;
    Y = labels;
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.5);
    for train_index,test_index in sss.split(data,data['readmitted']):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    train_l = train_set['readmitted'];
    test_l = test_set['readmitted'];
    train_f = train_set.drop(labels = ['readmitted'],axis=1);
    test_f = test_set.drop(labels = ['readmitted'],axis=1);

    dt = DecisionTreeClassifier(max_features=20);
    dt = dt.fit(train_f,train_l);

    pre_l = dt.predict(test_f);

    test_l_binarized = label_binarize(test_l,classes=[0,1,2]);
    pre_l_binarized = label_binarize(pre_l,classes=[0,1,2])

    print(accuracy_score(test_l_binarized,pre_l_binarized))
    print(avg_precision(test_l_binarized,pre_l_binarized))

if __name__ == "__main__":
    main()
