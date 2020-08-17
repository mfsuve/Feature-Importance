# * Feature importance functions have been gathered from;
# https://machinelearningmastery.com/calculate-feature-importance-with-python/
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from util import plot_feature_importances, fix_columns
from lgbm import lightGBM
import pandas as pd
import numpy as np
import argparse


def linear_regression(args, X, y):
    '''X needs to be a DataFrame with column names'''
    model = LinearRegression()
    scaled = StandardScaler().fit_transform(X)
    model.fit(scaled, y)
    # get importance
    importance = model.coef_
    # plot feature importance
    # (Took absolute value because if the coefficient is negative, there is still a correleation but reversed)
    plot_feature_importances(np.abs(importance), X.columns, 'linear_regression', args.folder)
    
    
def decision_tree(args, X, y):
    '''X needs to be a DataFrame with column names'''
    if args.type == 'regression':
        model = DecisionTreeRegressor()
    else:
        model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # plot feature importance
    plot_feature_importances(importance, X.columns, 'decision_tree', args.folder)
    

def random_forest(args, X, y):
    '''X needs to be a DataFrame with column names'''
    if args.type == 'regression':
        model = RandomForestRegressor()
    else:
        model = RandomForestClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # plot feature importance
    plot_feature_importances(importance, X.columns, 'random_forest', args.folder)


def permutation(args, X, y):
    '''X needs to be a DataFrame with column names'''
    if args.type == 'regression':
        model = KNeighborsRegressor()
        scoring = 'neg_mean_squared_error'
    else:
        model = KNeighborsClassifier()
        scoring = 'balanced_accuracy'
    # fit the model
    model.fit(X, y)
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring=scoring)
    # get importance
    importance = results.importances_mean
    # plot feature importance
    plot_feature_importances(importance, X.columns, 'permutation', args.folder)


def main(args):
    try:
        X = pd.read_csv(args.X)
        X = fix_columns(X)
        y = X.iloc[:, args.target]
    except FileNotFoundError:
        print(f'Error: {args.X} does not exists.')
        exit()
    except IndexError:
        print(f'Error: {args.target} for target column number is out of bounds of data from {args.X} with shape {X.shape}')
        exit()
    X = X.drop(X.columns[args.target], axis=1)
    
    try:
        if args.lgbm or args.all:
            lightGBM(args, X, y)
        if args.lr or args.all:
            linear_regression(args, X, y)
        if args.dt or args.all:
            decision_tree(args, X, y)
        if args.rf or args.all:
            random_forest(args, X, y)
        if args.perm or args.all:
            permutation(args, X, y)
    except ValueError as e:
        print(f'Error: {str(e).capitalize()}.')
        print('=> You need to preprocess you data before using this module.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculating and Plotting Feature Importances')
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-x', '--X', type=str, help='Path for a csv file to use as data', required=True)
    requiredNamed.add_argument('-t', '--target', type=int, help='Column number of the data to treat as labels (Negative values are supported to count from the end)', required=True)
    
    parser.add_argument('-A', '--all', action='store_true', help='Plot for all methods (overrides [-l, -r, -d, -R, -p])')
    parser.add_argument('-l', '--lgbm', action='store_true', help='Plot for LightGBM Feature Importance')
    parser.add_argument('-r', '--lr', action='store_true', help='Plot for Linear Regression Feature Importance')
    parser.add_argument('-d', '--dt', action='store_true', help='Plot for Decision Tree Feature Importance')
    parser.add_argument('-R', '--rf', action='store_true', help='Plot for Random Forest Feature Importance')
    parser.add_argument('-p', '--perm', action='store_true', help='Plot for Permutation Feature Importance')
    
    parser.add_argument('-T', '--type', choices=['regression', 'binary', 'multiclass'], default='binary', help='Type of the operation (default: "binary")')
    parser.add_argument('-f', '--folder', type=str, help='Folder name for plot to be saved (Optional, Creates the folder if it doesn\'t exist)')
    args = parser.parse_args()
    
    main(args)
    
