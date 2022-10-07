#%%
from Code import reader
#%%
if __name__ == '__main__':
    irtm = reader.read_dir("Assignments Datasets/IRTM")
    irtm.save_docs("Assignments Datasets/irtm")