# Feature Importance

A simple python module for investigating the feature importances for a given data.

Usage can be seen by;

    >> python feature_importance.py -h
   
    usage: feature_importance.py [-h] -x X -t TARGET [-A] [-l] [-r] [-d] [-R] [-p]
                                 [-T {regression,binary,multiclass}] [-f FOLDER]

    Calculating and Plotting Feature Importances

    optional arguments:
      -h, --help            show this help message and exit
      -A, --all             Plot for all methods (overrides [-l, -r, -d, -R, -p])
      -l, --lgbm            Plot for LightGBM Feature Importance
      -r, --lr              Plot for Linear Regression Feature Importance
      -d, --dt              Plot for Decision Tree Feature Importance
      -R, --rf              Plot for Random Forest Feature Importance
      -p, --perm            Plot for Permutation Feature Importance
      -T {regression,binary,multiclass}, --type {regression,binary,multiclass}
                            Type of the operation (default: "binary")
      -f FOLDER, --folder FOLDER
                            Folder name for plot to be saved (Optional, Creates
                            the folder if it doesn't exist)

    required arguments:
      -x X, --X X           Path for a csv file to use as data
      -t TARGET, --target TARGET
                         Column number of the data to treat as labels (Negative
                         values are supported to count from the end)
