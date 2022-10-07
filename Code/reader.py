#%%
import os
import requests
import pandas as pd
import pickle
from .__init__ import Document, Documents_Set
#%%
def read_doc(file_path:str, docID:int=None):
    init_token_list = []
    ## vertify type of input
    if not isinstance(file_path, str):
        print("you should input where you store your document in string type.")
        return False
    ## make document in to a list of list of strings, seperated in lines
    file_path = file_path.strip("/")
    document_list = open(file_path, 'rt').readlines()

    ## make document into a single list of string
    for line in document_list:
        start_flag = 0
        for stop_flag in range( len(line)):
            if line[ stop_flag] == ' ' :
                init_token_list.append( ''.join(line[ start_flag : stop_flag]))
                start_flag = stop_flag + 1
            # check the last word
            if line[-1] == '.' and stop_flag == len(line)-1: 
                init_token_list.append( ''.join(line[ start_flag : -1]))
    return Document(init_token_list, docID)
#%%
## read a set of documents in a dir
def read_dir(dir_path):
    if not isinstance(dir_path, str):
        print("you should input where you store your document in string type.")
        return False
    ## make document in to a list of list of strings, seperated in lines
    dir_path = dir_path.strip("/")
    
    docs = Documents_Set()
    for doc in os.listdir(dir_path):
        docID = int(doc[:-4])
        docs.add_doc(read_doc(file_path=(dir_path + "/" + doc), docID=docID))
    docs.finish_doc_adding()
    return docs
#%%
def load_docs(file_path:str)->Documents_Set:
    file_path = file_path.strip("/")
    with open(file_path, 'rb') as f:
        docs = pickle.load(f)
    return docs
#%%
def get_training_docIDs(url:str, save:bool=False):
    r = requests.get(url)
    all_text = r.text
    
    if save:
        with open("temp.txt", "w") as text_file:
            text_file.write(str(all_text))
        text_file.close()
        
    cats_list = all_text.split("\n")
    for i in range(len(cats_list)):
        cats_list[i] = cats_list[i].split()
        if "\r" in cats_list[i]:
            cats_list[i].pop[-1]

    training_dict = {}  # {cat:[docID]}
    for cat_list in cats_list:
        training_dict[cat_list[0]] = [int(docID) for docID in cat_list[1:]]
    return training_dict
#%%
## unit test
if __name__ == "__main__":
    print("unit test for read_doc()")
    pa1 = read_doc("Assignments Datasets/pa1.txt")
    print(pa1.init_tokens)
    print(pa1.docID)
    
    print("\n\nunit test for read_dir()")
    temp = read_dir("Assignments Datasets/IRTM")
    doc1 = temp.get_doc(10)
    print(doc1.docID)
    print(doc1.init_tokens)
#%%
