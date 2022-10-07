#%%
from Code import reader, constant, classifying
from Code.__init__ import Simple_Documents_Set
import pandas as pd
import numpy as np
#%%
if __name__ == '__main__':
    ## unit test
    irtm = reader.load_docs("Assignments Datasets/irtm.pickle")
    #%%
    url = "https://ceiba.ntu.edu.tw/course/88ca22/content/training.txt"
    training_docIDs = reader.get_training_docIDs(url)
    #%%
    ## train docs
    train_docs = Simple_Documents_Set(docs_list=[irtm.get_doc(docID) for cat in training_docIDs for docID in training_docIDs[cat]])
    #%%
    pa3 = classifying.NB_multi_Classifier()
    pa3.train(training_docIDs=training_docIDs, train_docs=train_docs)
    #%%
    ## unit tesst for predict
    # all_result = pa3.predict(Simple_Documents_Set(docs_list=[irtm.get_doc(docID) for docID in irtm.docIDs]))
    #%%
    ## split test from all docs
    all_train_docIDs = [docID for cat in training_docIDs for docID in training_docIDs[cat]]
    test_docIDs = []
    for docID in irtm.docIDs:
        if docID in all_train_docIDs:
            continue
        test_docIDs.append(docID)
    #%%
    ## prediction for pa3
    pa3_result = pa3.predict(Simple_Documents_Set(docs_list=[irtm.get_doc(docID) for docID in test_docIDs]))
    #%%
    ## save prediction
    file_name = "pa3.csv"
    pa3_result.to_csv(file_name, index=False)
    print(file_name, " saved")