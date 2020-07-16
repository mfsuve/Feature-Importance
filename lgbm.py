import lightgbm as lgb
import numpy as np
import pandas as pd
try:
    from util import plot_feature_importances
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    from util import plot_feature_importances
import os


def lightGBM(args, X, y, num_boost_round=300):
    '''X needs to be a Dataframe with column names
    
    Currently, no preprocessing is performed. Preprocessing for continuous and categorical features can be added in the future.
    
    lgb_params can be fine-tuned to get optimal results'''

    objective = args.type
    dtrain = lgb.Dataset(X, y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': objective,
        'learning_rate': 0.003,
        'tree_learner': 'voting',
    #     'boosting_type': 'rf',
    #     'subsample': 0.623,
    #     'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 20,
    #     'seed': None,
        'bagging_freq': 1,
        'bagging_fraction': .8,
        'n_jobs': -1,
        'verbose': -1
    }
    if objective == 'multiclass':
        lgb_params['num_class'] = np.unique(y).size
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=300)
    
    if args is None:
        plot_feature_importances(clf.feature_importance(), X.columns, 'lgbm')
    else:
        plot_feature_importances(clf.feature_importance(), X.columns, 'lgbm', args.folder)
    

if __name__ == '__main__':
    print('Working on iris dataset. To use this with some other data, please work with feature_importance.py.')
    from sklearn import datasets
    iris = datasets.load_iris()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = iris.target
    lightGBM(None, X, y)
