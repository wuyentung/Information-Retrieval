#%%
from operator import index
import numpy as np
import pandas as pd
from . import constant
#%%
def cal_idf(N:int, doc_freq_t:int):
    ## doc_freq_t: The number of documents in the collection that contain a given term t
    ## The idf of a rare term is high, and is likely to be low for a frequent term.
    return np.log10(N / doc_freq_t)
#%%
def make_doc_freq_df(df_dict:dict)->pd.DataFrame:
    
    df = pd.DataFrame.from_dict(df_dict, orient="index", columns=[ constant.DF]).sort_index()
    df[constant.TERM] = df.index
    df.reset_index(drop=True, inplace=True)
    df[constant.T_INDEX] = df.index.to_numpy()+1
    
    return df[[constant.T_INDEX, constant.TERM, constant.DF]]
#%%
def make_doc_tf_idf_df(df:pd.DataFrame, docID:int, doc_tf:dict)->pd.DataFrame:
    df_copy = df.copy(deep=True)
    df_copy[constant.T_INDEX] = np.arange(df_copy.shape[0])+1
    for term in df_copy.index:
        if term in doc_tf:
            continue
        df_copy.drop(axis=1, index=term, inplace=True)
    df_copy[constant.TF_IDF] = df_copy[str(docID)]
    return df_copy[[constant.T_INDEX, constant.TF_IDF]]