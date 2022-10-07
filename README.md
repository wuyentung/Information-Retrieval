# Information Retrieval Algorithms (2020_fall_IR)
This repository is the integrated version of algorithms and programming assignments (PAs) from Information Retrieval course.  
- Requirements: 
    - Install [NLTK](https://www.nltk.org/install.html) for stemming.
    - After cloning the repository, first unzip the file Assignment Data/IRTM.zip to let program able to read, and then run irtm.py getting pickle of irtm data.
- The following topics (module_name) describes the programming assignments.

## Programming Assignment 1 (pa1.py): Terms Extraction
Extract informative terms from a document ( *pa1.txt* in this module).
### Feature
- Tokenization.
- Lowercasing everything.
- Stemming using [Porter’s algorithm](https://tartarus.org/martin/PorterStemmer/).
- Stopword removal.
- Save the result as a txt file.

## Programming Assignment 2 (pa2.py): tf-idf
Convert a set of documents (1095 news documents) into tf-idf vectors.
### Feature
1. Construct a dictionary based on the terms extracted from the given documents.
    - Record the document frequency of each term.
    - Save as *dictionary.txt*
1. Transfer each document into a tf-idf unit vector. 
    - ${idf}_{t}=\log_{10}{\frac{N}{{df}_t}}$
    - Save as *DocID.txt*.
1. Write a function cosine($Doc_x$, $Doc_y$) which loads the tf-idf vectors of documents x and y and returns their cosine similarity.

## Programming Assignment 3 (pa3.py): Multinomial NB Classifier
Multinomial Naive Base (NB) Classifier for 1095 documents.
### Feature
- 13 classes (id 1~13), each class has 15 training documents
    - https://ceiba.ntu.edu.tw/course/88ca22/content/training.txt
- The remaining documents are for testing.
    - https://www.kaggle.com/competitions/2020irtmhw3

- Employ feature selection method and use only 500 terms in the classification.
    - $Χ^2$ (chi-squared) test.
    - Likelihood ratio.
    - Hybrid $Χ^2$ test and likelihood ratio.
    - reason for feature selection:
        - For each class, calculate M P(X=t|c) parameters, where M is the size of the vocabulary.
        - Then the total number of parameters in the system will be |C|*M &larr; can be a huge number.
        - Many terms in the vocabulary are not indicative.

- When classify a testing document, ignore terms that does not in the selected vocabulary.
- To avoid zero probabilities, calculate $P(X=t|c)$ by using add-one smoothing.
$$P(X=t|c)=\frac{T_{ct_k}+1}{\sum_{t'\in V}{(T_{ct'}+1)}}=\frac{T_{ct_k}+1}{\sum_{t'\in V}{(T_{ct'})}+|V|}$$

## Programming Assignment 4 (pa4.py): HAC clustering
Hierarchical Agglomerative Clustering (HAC) clustering for 1095 documents.
### Feature
- Documents are represented as normalized tf-idf vectors.
- Cosine similarity for pair-wise document similarity.
    - Single-link clustering
- Implement own HEAP to speed up clustering from $O(N^3)$  to  $O(N^2 \log{N})$
- Cluster result: *K*.txt, 
    - For each cluster, doc_id are ordered ascending.
    - Clusters are separated by an empty line.

## Final Group Project: Emoji Recommender
Project goal: emoji prediction using content based data  
- Two stage training -- clustering and NB multinomial classifying
- Embedding -- vote for two NB multinomial classifier
- More project detail in IR_21組.pptx