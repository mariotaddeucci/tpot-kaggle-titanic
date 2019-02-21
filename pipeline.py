import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.831453071833957
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.7500000000000001)),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.35000000000000003, n_estimators=100), step=0.7000000000000001),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=14, min_samples_split=17, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
