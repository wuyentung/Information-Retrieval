#%%
from Code import reader, constant
import pandas as pd
import numpy as np
#%%
if __name__ == '__main__':
    ## unit test
    # temp = reader.read_dir("Archieve/PA2/emp")
    #%%
    # temp.save_doc_freq()
    # %%
    # temp.save_doc_tf_idf(1)
    
    pa2 = reader.read_dir("Assignments Datasets/IRTM")
    #%%
    # print(pa2.get_doc(1).terms)

    ## part 1
    pa2.save_doc_freq()

    ## part 2
    pa2.save_doc_tf_idf(1)
    #%%
    ## part 3 : cosine similarity between document 1 and 2 
    print(pa2.get_single_similarity(1, 2, normed=False))


#%%
