from . import constant, preprocessing, detail_calculation
import pandas as pd
import numpy as np
import pickle
#%%
'''
terms_dict in Document:
{term: term frequency}
'''
class Term():
    def __init__(self, term:str):
        self.term = term
        self.docID_list = []
        pass
#%%
class Document():
    def __init__(self, init_tokens:list, docID:int):
        self.init_tokens = init_tokens
        self.docID = docID
        self.terms = preprocessing.Preprocessor().extract_terms(self.init_tokens)
        pass

    def save_txt(self, target:str=constant.TERMS, file_name:str="result.txt"):
        if constant.TERMS == target:
            content = self.terms
        elif constant.INIT_TOKENS == target:
            content = self.init_tokens
        with open(file_name , "w") as text_file:
            text_file.write(str(content))
        print("file saved")
        
    def set_term_frequencies(self, term_frequencies:dict):
        ## {term: term frequency}
        self.term_frequencies = term_frequencies
        pass
#%%
class Simple_Documents_Set():
    def __init__(self, docs_list:list):
        self.docs_dict = {doc.docID:doc for doc in docs_list}
        pass     
    def get_doc(self, docID:int)->Document:
        return self.docs_dict[docID]
    def get_all_docs(self)->list:
        return [self.docs_dict[docID] for docID in self.docs_dict]
    def __nothing():
        pass
#%%
class Documents_Set():
    def __init__(self,):
        self.docs_dict = {}
        self.docIDs = []
        pass
    
    def save_docs(self, file_name:str):
        with open(f'{file_name}.pickle', 'wb') as file:
            pickle.dump(self, file)
    
    def add_doc(self, doc:Document):
        self.docs_dict[doc.docID] = doc
        self.docIDs.append(doc.docID)
        pass
    
    def finish_doc_adding(self):
        self.docIDs = sorted(self.docIDs)
        print("Finish doc adding and term extracting, now calculate term-document matrix and similarity matrix.\n-")
        self.N = len(self.docIDs)
        self._cal_doc_freq()
        print("Finish document frequencies calculation.\n-")
        self._cal_term_doc_martix()
        print("Stored tf-idf calculation in term-document matrix.\n-")
        self._cal_similarity_matrix()
        print("Similarity martix is calculated.\n-")
        pass
    
    def get_doc(self, docID:int)->Document:
        return self.docs_dict[docID]
    
    ## tf-idf related
    def _cal_doc_freq(self,):
        ## {term: document frequency}
        self.document_frequencies = _cal_doc_freq(self)
        self.doc_freq_df = detail_calculation.make_doc_freq_df(self.document_frequencies)
        pass
    def save_doc_freq(self, file_name:str="dictionary.txt"):
        with open(file_name , "a") as text_file:
            text_file.write(self.doc_freq_df.to_string(index=False))
        return f"{file_name} saved"
    
    def _cal_term_doc_martix(self):
        self.term_doc_martix = _cal_term_doc_martix(self)
        self.norm_term_doc_martix = _cal_term_doc_martix(self, normed=True)
        pass
    def save_doc_tf_idf(self, docID, file_name=""):
        if 0 == len(file_name):
            file_name = str(docID) + ".txt"
        doc_tf_idf_df = detail_calculation.make_doc_tf_idf_df(df=self.term_doc_martix, docID=docID, doc_tf=self.get_doc(docID).term_frequencies)
        with open(file_name , "a") as text_file:
            text_file.write(str(len(self.get_doc(docID).term_frequencies))+"\n")
            text_file.write(doc_tf_idf_df.to_string(index=False))
        return f"{file_name} saved"
    
    ## document similarity related
    def _cal_similarity_matrix(self,):
        self.similarity_matrix = _make_similarity_matrix(self.term_doc_martix)
        self.norm_similarity_matrix = _make_similarity_matrix(self.norm_term_doc_martix)
        pass
    def save_similarity_matrix(self, file_name:str="similarity.csv", normed:bool=True):
        target = self.norm_similarity_matrix if normed else self.similarity_matrix
        pd.DataFrame(target).to_csv(file_name, index=False, header=False)
        return f"{file_name} saved"
    def get_single_similarity(self, docID_1:int, docID_2:int, normed:bool=False):
        target = self.norm_similarity_matrix if normed else self.similarity_matrix
        return target[docID_1-1, docID_2-1]
#%%
def _cal_doc_freq(docs:Documents_Set):
    document_frequencies = {}
    ## {term(str): df(int)}
    for docID in docs.docIDs:
        doc = docs.get_doc(docID)
        term_frequencies = {}
        ## {term(str): tf(int)} // for each doc
        for term in doc.terms:
            if term in document_frequencies:
                if term in term_frequencies:
                    term_frequencies[term] += 1
                else:
                    term_frequencies[term] = 1
                    document_frequencies[term] += 1
            else:
                document_frequencies[term] = 1
                term_frequencies[term] = 1
                
            doc.set_term_frequencies(term_frequencies)
    return document_frequencies
#%%
def _cal_term_doc_martix(docs:Documents_Set, normed:bool=False):
    terms = docs.doc_freq_df[constant.TERM].to_list()
    sorted_docIDs = sorted(docs.docIDs[:])
    term_doc_dict = {}
    for term in terms:
        term_tf_idf_list = []
        # print(term)
        for docID in sorted_docIDs:
            if term in docs.get_doc(docID).term_frequencies:
                tf = docs.get_doc(docID).term_frequencies[term]
            else:
                tf = 0
            idf = np.log10(docs.N / docs.document_frequencies[term])
            if normed:
                idf = np.log10((1+docs.N) / (1+docs.document_frequencies[term])) + 1
            # print(f"\t{tf}")
            # print(f"\t{docs.N}")
            # print(f"\t{docs.document_frequencies[term]}")
            # print(f"\t{idf}")
            ## tf-idf adding
            term_tf_idf_list.append(tf * idf)
        term_doc_dict[term] = term_tf_idf_list
        # if "abc" == term:
        #     break
    return pd.DataFrame.from_dict(term_doc_dict, orient="index", columns=[str(docID) for docID in sorted_docIDs])
#%%
def _make_similarity_matrix(term_doc_matrix:pd.DataFrame)->np.ndarray:
    term_doc_matrix = term_doc_matrix.to_numpy()
    def sim(docID_1, docID_2):
        docu_1_vec = term_doc_matrix[:, docID_1]
        docu_2_vec = term_doc_matrix[:, docID_2]

        ## Dot and norm
        dot = np.sum(np.dot(docu_1_vec, docu_2_vec))
        norm_a = np.sqrt(np.sum(np.square(docu_1_vec)))
        norm_b = np.sqrt(np.sum(np.square(docu_2_vec)))

        # Cosine similarity
        similarity = dot / (norm_a * norm_b)
        return similarity
    n = term_doc_matrix.shape[1]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = sim(i, j)
    return matrix