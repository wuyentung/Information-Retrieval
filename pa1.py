#%%
from Code import reader, preprocessing
#%%
if __name__ == "__main__":
    ## Create an object of TermExtractor
    pa1 = reader.read_doc('Assignments Datasets/pa1.txt')
    #%%
    
    # you can save your own document named "result.txt" by .save_result()
    pa1.save_txt()