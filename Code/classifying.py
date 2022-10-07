#%%
from operator import index
import pandas as pd
import numpy as np
from copy import deepcopy
import math

from .__init__ import Document, Documents_Set, Simple_Documents_Set
from . import preprocessing, constant
#%%
class NB_multi_Classifier():
    def __init__(self):
        self.categories = [] #[cat]
        self.selected_terms = [] #[term]
        ## In pa3, all prior of categories is the same
        self.priors = {} #{cat:prior}
        self.terms_cond_probs = {} #{term:{cat:cond_prob}}
        pass
    def train(self, training_docIDs:dict, train_docs:Simple_Documents_Set, feature_selection_methid:str="sum_likelihood"):
        self.categories = list(training_docIDs.keys())
        self._feature_selection(training_docIDs=training_docIDs, train_docs=train_docs, method=feature_selection_methid)
        self.priors, self.terms_cond_probs = _cal_condprob(training_docIDs=training_docIDs, train_docs=train_docs, selected_terms=self.selected_terms)
        pass

    def _feature_selection(self, training_docIDs:dict, train_docs:Simple_Documents_Set, method="sum_likelihood"):
        self.terms_nx = _terms_nx_making(training_docIDs, train_docs)
        self.selected_terms = _selection(self.terms_nx, method)
        pass
    
    def predict(self, test_docs:Simple_Documents_Set):
        predictions = []
        docIDs = []
        for doc in test_docs.get_all_docs():
            docIDs.append(doc.docID)
            predictions.append(_apply_multi_NB(catigories=self.categories, prior=self.priors, selected_terms=self.selected_terms, terms_cond_probs=self.terms_cond_probs, test_doc=doc))
        print("prediction made")
        result = pd.DataFrame()
        result["Id"] = docIDs
        result["Value"] = predictions
        return result
#%%

def _terms_nx_making(training_docIDs:dict, train_docs:Simple_Documents_Set):
    
    terms_nx = {} #{term: df of cat and nx}
    cats_n_doc = [len(training_docIDs[cat]) for cat in training_docIDs]
    ## create empty terms_present for each term
    for doc in train_docs.get_all_docs():
        for term in doc.term_frequencies:
            if term not in terms_nx:
                ## so far {term: df of cat and present/absent}
                terms_nx[term] = pd.DataFrame(np.zeros((len(training_docIDs), 1)), index=training_docIDs.keys(), columns=[constant.PRESENT])
                terms_nx[term][constant.ABSENT] = cats_n_doc
    ## fill terms_present for each term
    for cat in training_docIDs:
        for doc in [train_docs.get_doc(docID) for docID in training_docIDs[cat]]:
            for term in doc.term_frequencies:
                terms_nx[term].loc[cat, constant.PRESENT] += 1
                terms_nx[term].loc[cat, constant.ABSENT] -= 1
                
    for term in terms_nx:
        ## filling nx
        terms_nx[term] = _filling_nx(nx=terms_nx[term])
        ## get likelihood ratio, chi square for each term
        terms_nx[term] = _cal_ratio(nx=terms_nx[term])
    return terms_nx

def _filling_nx(nx:pd.DataFrame,):
    # get matrix for each term
    # return modified matrix
    nx[constant.N11] = nx[constant.PRESENT]
    nx[constant.N10] = nx[constant.PRESENT]
    nx[constant.N01] = [nx[constant.PRESENT].sum() - nx[constant.N11][cat] for cat in nx.index]
    nx[constant.N00] = [nx[constant.ABSENT].sum() - nx[constant.N10][cat] for cat in nx.index]

    return nx
#%%
def _cal_ratio(nx:pd.DataFrame):
    # return likelihood ratio, chi square of term in all cat
    likelihoods = []
    chi_squares = []
    for cat in nx.index:
        n11 = nx["n11"][cat]
        n10 = nx["n10"][cat]
        n01 = nx["n01"][cat]
        n00 = nx["n00"][cat]
        N = n11 + n10 + n01 + n00

        ## likelihood ratio
        pt = (n11 + n01) / N
        p1 = n11 / (n11 + n10)
        p2 = n01 / (n01 + n00)

        likelihoods.append(-2 * np.log10((((pt ** n11) * (1 - pt) ** n10) * ((pt ** n01) * (1 - pt) ** n00)) / (((p1**n11) * (1 - p1)**n10) * ((p2**n01) * (1 - p2)**n00))))

        ## chi square
        e11 = (n11 + n01) * (n11 + n10) / N
        e10 = (n11 + n10) * (n00 + n10) / N
        e01 = (n11 + n01) * (n01 + n00) / N
        e00 = (n00 + n01) * (n00 + n10) / N

        chi_squares.append((n11 - e11)**2 / e11 + (n10 - e10)**2 / e11 + (n01 - e01)**2 / e11 + (n00 - e00)**2 / e11)
    nx[constant.LIKELIHOOD_RATIO] = likelihoods
    nx[constant.CHI_SQR] = chi_squares

    return nx

#%%
def test_term_extract(all_doc, training_df):
    temp = []
    for index, row in training_df.iterrows():
        temp.append(row.to_list())
    train_ids = []
    for row in temp:
        for item in row:
            train_ids.append(item)

    dict_dict = {} ## key:doc_id(in real), values:preprocess_list
    for index in range(len(all_doc.preprocess_list)):
        if index in train_ids:
            continue
        dict_dict[index +1] = all_doc.preprocess_list[index].word_dic
        # use .word_dic to get each word
    return dict_dict
#%%

def _selection(terms_nx:dict, method="sum_likelihood"):
    # terms_nx {term:df of cats}
    store_score = {}
    selected_terms = []
    feature_limit = 500

    if method == "sum_likelihood":
        ## top 500 from sum of the likelihood ratio
        for term in terms_nx.keys():
            store_score[term] = terms_nx[term][constant.LIKELIHOOD_RATIO].sum()
        selected_terms = sorted(store_score.items(), key=lambda item: item[1], reverse=True)
        for rank in range(feature_limit):
            selected_terms.append(selected_terms[rank][0])

    if method == "max_likelihood":
        ## top 500 from the likelihood ratio
        for term in terms_nx.keys():
            store_score[term] = terms_nx[term][constant.LIKELIHOOD_RATIO].max()
        selected_terms = sorted(store_score.items(), key=lambda item: item[1], reverse=True)
        for rank in range(feature_limit):
            selected_terms.append(selected_terms[rank][0])

    if method == "max_chi":
        ## top 500 from the chi_sqr ratio
        for term in terms_nx.keys():
            store_score[term] = terms_nx[term][constant.CHI_SQR].max()
        selected_terms = sorted(store_score.items(), key=lambda item: item[1], reverse=True)
        for rank in range(feature_limit):
            selected_terms.append(selected_terms[rank][0])

    if method == "hy_max":
        ## top 500 from the hybrid ratio
        store_score_like = {}
        store_score_chi = {}
        for term in terms_nx.keys():
            store_score_like[term] = terms_nx[term][constant.LIKELIHOOD_RATIO].max()
            store_score_chi[term] = terms_nx[term][constant.CHI_SQR].max()
        rank_like = sorted(store_score_like.items(), key=lambda item: item[1], reverse=True)
        rank_chi = sorted(store_score_chi.items(), key=lambda item: item[1], reverse=True)
        buffer = set()
        def hybrid_rank(check, rank_list, term):
            if term in check:
                check.discard(term)
                rank_list.append(term)
            else:
                check.add(term)

        for i in range(len(rank_chi)):
            hybrid_rank(buffer, selected_terms, rank_chi[i][0])
            hybrid_rank(buffer, selected_terms, rank_like[i][0])
            if len(selected_terms) == feature_limit:
                break

    print("feature_selection with %s completed" %method)
    return selected_terms
#%%
def _cal_condprob(training_docIDs:dict, train_docs:Simple_Documents_Set, selected_terms:list):
    n_cat = len(training_docIDs)
    terms_counts = {} # {term:{cat:value}}
    terms_cond_probs = {} # {term:{cat:value}}
    priors = {} # {cat:value}
    for term in selected_terms:
        terms_counts[term] = {cat:0 for cat in training_docIDs}
        terms_cond_probs[term] = {}
    for cat in training_docIDs:
        priors[cat] = len(training_docIDs[cat]) / len(train_docs.docs_dict)
        for doc in [train_docs.get_doc(docID) for docID in training_docIDs[cat]]:
            for term in doc.term_frequencies:
                if term in selected_terms:
                    terms_counts[term][cat] += doc.term_frequencies[term]
        for term in selected_terms:
            ## add-one smoothing 
            terms_cond_probs[term][cat] = (int(terms_counts[term][cat]) + 1) / (sum([terms_counts[term][cat] for cat in terms_counts[term]]) - terms_counts[term][cat] + len(selected_terms))
            
    print("Calculate conditional probability completed (training complete).")
    return priors, terms_cond_probs
#%%
def _apply_multi_NB(catigories:list, prior:dict, selected_terms:list, terms_cond_probs:dict, test_doc:Document):
    hit_terms = []
    score = {}
    for term in test_doc.term_frequencies:
        if term in selected_terms:
            hit_terms.append(term)
    for cat in catigories:
        score[cat] = math.log10(prior[cat])
        for term in hit_terms:
            score[cat] += math.log10(terms_cond_probs[term][cat])
    prediction = catigories[0]
    for cat in catigories:
        if score[cat] >= score[prediction]:
            prediction = cat
    return prediction ## catigory for single document

#%%
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

















