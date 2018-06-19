from keras.preprocessing import image
from PIL import Image
import keras.applications.inception_v3 as inception_v3

from sklearn import metrics, preprocessing, feature_selection
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split


from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords


import lightgbm as lgb
import numpy as np
import pandas as pd
import math
import os
import gc
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--valid',  dest='VALID', action='store_true')
parser.add_argument('--no-valid',  dest='VALID', action='store_false')
parser.set_defaults(VALID=False)
args = parser.parse_args()

print('valid is ', args.VALID)

training = pd.read_csv('./data/train.csv', index_col = 'item_id', parse_dates = ['activation_date'])
traindex = training.index
testing = pd.read_csv('./data/test.csv', index_col = 'item_id', parse_dates = ['activation_date'])
testdex = testing.index

y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print("Train shape: {} Rows, {} Columns".format(*training.shape))
print("Test shape: {} Rows, {} Columns".format(*testing.shape))
ntrain = training.shape[0]
ntest = testing.shape[0]

image_files = [x.path for x in os.scandir('./data/images/')]
inception_model = inception_v3.InceptionV3(weights='imagenet')

def classify_inception(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
    except (OSError, IOError):
        return [0,0,0.5]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    preds = inception_model.predict(x)
    return inception_v3.decode_predictions(preds, top=1)[0][0]

def image_id_from_path(path):
    return path.split('/')[3].split('.')[0]

inception_conf = [[image_id_from_path(x), classify_inception(x)[2]] for x in image_files]
confidence = pd.DataFrame(inception_conf, columns=['image', 'image_confidence'])
confidence.to_csv('./save_encoded_img.csv')

df = pd.concat([training, testing], axis=0)
del training, testing
gc.collect()
df['price'] = np.log(df['price']+0.0001)
df["price"].fillna(df.price.mean(),inplace=True)
df['weekday'] = df['activation_date'].dt.weekday
df['weekn of year'] = df['activation_date'].dt.week
df['dayn of month'] = df['activation_date'].dt.day
df["image_top_1"].fillna(-999,inplace=True)

training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index

df.drop('activation_date', axis=1, inplace=True)

categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('missing')
    df[col] = lbl.fit_transform(df[col].astype(str))

textfeats = ["description", "title"]

for cols in textfeats:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}
def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
           #max_features=16000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            #max_features=7000,
            preprocessor=get_col('title')))
    ])

vectorizer.fit(df.loc[traindex,:].to_dict('records'))
fitted_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
df.drop(["description", "title"], axis=1,inplace=True)

kf = KFold(n_splits=4, shuffle=True, random_state=42)

def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((4, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if seed_bool == True:
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

ridge_params = {
    'alpha': 20.0,
    'fit_intercept':True,
    'normalize':False,
    'copy_X':True,
    'max_iter':None,
    'tol':0.001,
    'solver':'auto',
    'random_state': 42
}

ridge = SklearnWrapper(clf=Ridge, seed = 42, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, fitted_df[:ntrain], y, fitted_df[ntrain:])
rms = math.sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds'] = ridge_preds
df_confidence = pd.merge(df.reset_index(), confidence, how='left', on='image').set_index('item_id')
## start to create train data
df_confidence.drop("image", axis=1,inplace=True)
X = hstack([csr_matrix(df_confidence.loc[traindex,:].values),fitted_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df_confidence.loc[testdex,:].values),fitted_df[traindex.shape[0]:]])
tfvocab = df_confidence.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df, confidence, df_confidence
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=23)

lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 250,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.85,
    # 'bagging_freq': 5,
    'learning_rate': 0.02,
    'verbose': 0
}

if args.VALID == True:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23)

    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X, X_train
    gc.collect()

    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=n_rounds,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print("Model Evaluation Stage")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    del X_valid ; gc.collect()

else:
    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(X, y,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X
    gc.collect()
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1000,
        verbose_eval=100
    )

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)
