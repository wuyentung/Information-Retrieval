#%%
import numpy as np
from copy import deepcopy
from .__init__ import Document, Documents_Set, Simple_Documents_Set
from . import heap, constant
#%%
class Efficient_HAC():
    def __init__(self, similarities:np.ndarray):
        self.N = similarities.shape[0]
        ## C[i][j]: the similarity between clusters i and j.
        ## C[i][j]["sim"] -> value of similarity with i and j
        ## C[i][j]["index"] -> the index j for i to calculate similarity
        self.C = [[{constant.SIMILARITY:similarities[i][j], constant.J_INDEX:j} for j in range(similarities.shape[1])] for i in range(similarities.shape[0])]
        ## I: a list to indicate which clusters are still available to be merged.
        self.I = [1 for _ in range(self.N)]
        ## P: a list of priority queue.
        self.P = [heap.Similarity_Heap(i, self.C[i]) for i in range(self.N)]
        ## A: a list of merges.
        self.A = []
        pass
    
    ## find max similarity and merge clusters
    def do_cluster(self):
        for _ in range(self.N-1):
            ## k1 < k2, k2 merged into k1
            k1 = self.argmax_i()
            k2 = self.P[k1].get_max_item()[constant.J_INDEX]
            ## swap if k1 > k2
            if k1 > k2:
                temp_k = k1
                k1 = k2
                k2 = temp_k
            ## merge and update attributes
            self.A.append({k1, k2})
            self.I[k2] = 0
            self.P[k1].clear()
            
            ## update priority queue and similarity
            for i in range(self.N):
                if not self.I[i]:
                    continue
                if i == k1:
                    continue
                self.traced_i = i
                self.P[i].delet_item(self.C[i][k1])
                self.P[i].delet_item(self.C[i][k2])
                self.C[i][k1][constant.SIMILARITY] = self.C[k1][i][constant.SIMILARITY] = self._single_link(i, k1, k2)
                self.P[i].insert(self.C[i][k1])
                self.P[k1].insert(self.C[k1][i])
                # break
            # break
        # print("Finish HAC clustering with single-link.")
        pass
    def argmax_i(self):
        '''

        :param queue:
        :type: [{},{}]
        :return:
        :rtype: dict
        '''
        ## init a legal local max
        for i in range(self.N):
            if self.I[i]:
                local_max = self.P[i].get_max_item()
                local_max_index = i
                break
        ## find global max
        for i in range(self.N):
            if self.I[i]:
                if local_max[constant.SIMILARITY] < self.P[i].get_max_item()[constant.SIMILARITY]:
                    local_max = self.P[i].get_max_item()
                    local_max_index = i
        return local_max_index

    def _single_link(self, i:int, k1:int, k2:int):
        if self.C[i][k1][constant.SIMILARITY] <= self.C[i][k2][constant.SIMILARITY]:
            bigger_sim = self.C[i][k1][constant.SIMILARITY]
        else:
            bigger_sim = self.C[i][k2][constant.SIMILARITY]
        return bigger_sim
    def get_K_cluster(self, k:int)->dict:
        def nothing():
            pass
        crop_A = self.A[:-k+1]
        merge_dict = {i:[i] for i in range(self.N)}
        for merge in crop_A:
            merge_dict[merge[0]]+=(merge_dict[merge[1]])
            del merge_dict[merge[1]]
        for cluster in merge_dict.keys():
            merge_dict[cluster] = sorted(merge_dict[cluster])
        return merge_dict
#%%
if __name__ == '__main__':
    pass