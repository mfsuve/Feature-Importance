import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
from datetime import datetime


def get_curr_time_str():
    return datetime.now().strftime("%d.%m.%Y_%H.%M.%S")


def make_sure_folder_exists(pathname):
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
        print(f' * {pathname} folder is created.')
    
    
def plot_feature_importances(feature_importances, feature_names, method_name, save_folder=None):
    '''
    feature_importances needs to be a list or a 1D numpy array
    '''
    feature_imp = pd.DataFrame(sorted(zip(feature_importances, feature_names)), columns=['Importance Value', 'Features'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Importance Value", y="Features", data=feature_imp.sort_values(by="Importance Value", ascending=False))
    plt.title(f'{" ".join(method_name.split("_")).title()} Feature Importances')
    plt.tight_layout()
    filename = f'{method_name}_importances_{get_curr_time_str()}.png'
    if save_folder is None:
        pathname = filename
    else:
        make_sure_folder_exists(save_folder)
        pathname = os.path.join(save_folder, filename)
        
    plt.savefig(pathname)
    
    saved_text = f' * Saved {pathname}'
    if save_folder is not None:
        saved_text += f' to {save_folder} folder'
    saved_text += '! ';
    print('=' * len(saved_text))
    print(saved_text)
    print('=' * len(saved_text))
