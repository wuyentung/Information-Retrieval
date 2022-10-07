#%%
from nltk.stem import PorterStemmer 
#%%
class TermExtractor():
    def __init__(self):
        self.inittoken_list = []
        self.stemmed_list = []
        self.stopwarded_list = []
        self.punctuation_list = [".", "'", '"', "?"]
        self.poter = PorterStemmer()
        self.stopward_list = []
        self.stopward_dic = {"ALL":[]}
        self.stopward_flag = [0, 0]
        return None
    
    def read_file(self, storage_place):
        if len(self.inittoken_list) != 0:
            return "you have already put some data in here"
        ## vertify type of input
        if not isinstance(storage_place, str):
            print("you should input where you store your document in string type.")
            return False
        ## make document in to a list of list of strings, seperated in lines
        storage_place = storage_place.strip("/")
        document_list = open(storage_place, 'rt').readlines()

        ## make document into a single list of string
        for line in document_list:
            start_flag = 0
            for stop_flag in range( len(line)):
                if line[ stop_flag] == ' ' :
                    self.inittoken_list.append( ''.join(line[ start_flag : stop_flag]))
                    start_flag = stop_flag + 1
                if line[-1] == '.' and stop_flag == len(line)-1: # check the last word
                    self.inittoken_list.append( ''.join(line[ start_flag : -1]))
        return self.inittoken_list
    
    def stemming(self):
        if len(self.stemmed_list) != 0:
            return "you have already stemmed your document"
        for voca_index in  range( len( self.inittoken_list)):
            for pun in self.punctuation_list:
                if pun in self.inittoken_list[voca_index] :
                    self.inittoken_list[voca_index] = self.inittoken_list[voca_index].replace(pun, '')
        ## lowercast and stem the document, where poter.stem() will auto-lowercast
        for voca in self.inittoken_list:
            self.stemmed_list.append( self.poter.stem( voca))    
        return self.stemmed_list
    
    def stopwarding(self):
        ## check for if stopward list create or not
        if len(self.stopward_list) == 0:
            init_stopward_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
            self.stopward_adding( init_stopward_list)
        
        ## check for the state of stopward, same for stopwarded, different for not yet 
        if self.stopward_flag[0] == self.stopward_flag[1]:
            return "you have already stopwarded your document"
        else:
            if self.stopward_flag[1] > 0:
                dest_document = self.stopwarded_list
            else:
                dest_document = self.stemmed_list
        
        ## stopward
        for voca_index in range( len( dest_document)):
            if dest_document[voca_index] in self.stopward_dic:
                self.stopward_dic['ALL'].append( voca_index)
                self.stopward_dic[ dest_document[voca_index]].append(voca_index)
            else:
                self.stopwarded_list.append( dest_document[voca_index])
        self.stopward_flag[1] = self.stopward_flag[1] +1
        return self.stopwarded_list
    
    def stopward_adding(self, new_ward_list):
        ## check for type of list
        if not isinstance(new_ward_list, list):
            print("want a list. in stopward_adding")
            return False
        for stopward in new_ward_list:
            ## check for type of each ward in list
            if not isinstance(stopward, str):
                print("want a list of string. in stopward_adding")
                return False
            ## stem and add 
            stemmed_stopward = self.poter.stem( stopward)
            if not stemmed_stopward in self.stopward_dic:
                self.stopward_list.append( stemmed_stopward)
                self.stopward_dic.update({stemmed_stopward: []})
        self.stopward_flag[0] = self.stopward_flag[0] +1
        return 0
    
    def punctuation_adding(self, new_pun):
        
        return 0
    
    def save_result(self, file_name:str="result.txt"):
        with open(file_name , "w") as text_file:
            text_file.write(str(self.stopwarded_list))
        print("file saved")
        pass

#%%
if __name__ == "__main__":
    ## Create an object of TermExtractor
    temp1 = TermExtractor()

    # use .read_file(file_route) to import document you want
    temp1.read_file('pa1.txt')
    
    # use .stemming() to lowercast and stem your document
    temp1.stemming()
    
    # use .stopwarding() to stopward your document
    temp1.stopwarding()
    
    # you can save your own document named "result.txt" by .save_result()
    temp1.save_result()