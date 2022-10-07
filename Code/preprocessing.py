#%%
from nltk.stem import PorterStemmer 
from . import reader
#%%
class Preprocessor():
    def __init__(self):
        self.punctuation_list = [
            ".", "'", '"', "?", ",", ")", "(", "[", "]", "{", "}", "@", "%", "$", "*", "-", "_", "/", "!", "#", "^", "&", "`", ":", ";", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
            ]
        init_stopward_list = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "", 
            ]
        self.stopward_set = set(PorterStemmer().stem(stopward) for stopward in init_stopward_list)
        pass
    
    def extract_terms(self, init_tokens:list):
        extracted_terms = []
        expanded_tokens = _internet_address_expand(init_tokens)
        for token in expanded_tokens:
            pun_removed = self._punctuation_remove(token).lower()
            stemmed = PorterStemmer().stem(pun_removed)
            if stemmed in self.stopward_set:
                continue
            extracted_terms.append(stemmed)
        return extracted_terms
    
    def _punctuation_remove(self, init_token:str):
        for pun in self.punctuation_list:
            if pun in init_token :
                init_token = init_token.replace(pun, '')
        return init_token
#%%
def _internet_address_expand(init_tokens:list):
    expanded_tokens = init_tokens[:]
    for token in init_tokens:
        if "http" in token or "www" in token:
            expanded_tokens.remove(token)
            token = token.replace("/", ".")
            token = token.replace("-", ".")
            expanded_token = token.split(".")
            expanded_tokens += expanded_token
    return expanded_tokens
#%%
## unit test
if __name__ == "__main__":
    # print(PorterStemmer().stem("the"))
    pa1 = reader.read_doc("Assignments Datasets/pa1.txt")
    Preprocessor().extract_terms(pa1)
    pa1.save_txt()
#%%
