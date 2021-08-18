import pandas as pd
import numpy as np


PATH_TO_WIKI = './data/wikipedia.csv'
def re_bin_wikipedia(bin_size=1):
    df = pd.read_csv(PATH_TO_WIKI).reset_index()
    df.columns = ['user_id','item_id','timestamp','state_label',]+[f'feature_{x}' for x in range(172)]
    df['timestamp'] = pd.Series(df.index).apply(lambda x: x//bin_size)
    path_to_new = PATH_TO_WIKI.rpartition('.')[0]+f'_binsize={bin_size}.csv'
    df.to_csv(path_to_new,header=False, index=False)#['timestamp']
    return path_to_new