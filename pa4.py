#%%
from Code import reader, constant, clustering
from Code.__init__ import Simple_Documents_Set
import pandas as pd
import numpy as np
#%%
if __name__ == '__main__':
    irtm = reader.load_docs("Assignments Datasets/irtm.pickle")
    #%%
    pa4 = clustering.Efficient_HAC(similarities=irtm.norm_similarity_matrix)
    #%%
    pa4.do_cluster()
    #%%
    K = {8, 13, 20}
    for k in K:
        with open(f'{k}.txt', 'w') as f:
            for key, value in pa4.get_K_cluster(k).items():
                print(value, file=f)