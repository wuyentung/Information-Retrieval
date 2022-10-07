#%%
import numpy as np
from . import constant
#%%
class Similarity_Heap():
    def __init__(self, i:int, i_similarities:list):
        ## i_similarities[j][constant.SIMILARITY] -> value of similarity with i and j
        ## i_similarities[j][constant.J_INDEX] -> index j, for i to calculate similarity
        self.priority_queue = i_similarities[:]
        # self.n = i_similarities
        ## build priority queue
        self._build()
        ## don’t want self-similarities
        self.delet_item(i_similarities[i])
        pass
    def get_max_item(self)->dict:
        return self.priority_queue[0]
    def _adjust(self, parent_index, ):
        ''' adaptive heap operation

        :param stored_index: list of where the data stored
        :param queue: [0,...,n-1], each store {constant.SIMILARITY: , constant.J_INDEX: ,}
        :type queue: [{}, {}]
        :param i: parent node id
        :param n: size of tree
        :return: None
        '''
        n = len(self.priority_queue)
        x = self.priority_queue[parent_index]
        child_index = 2 * parent_index + 1
        while child_index <= n-1:
            ## compare children
            if child_index < n-1:
                if self.priority_queue[child_index][constant.SIMILARITY] < self.priority_queue[child_index + 1][constant.SIMILARITY]:
                    child_index+=1
            ## compare x and child
            if x[constant.SIMILARITY] >= self.priority_queue[child_index][constant.SIMILARITY]:
                break
            ## swap child to parent
            else:
                self.priority_queue[(child_index-1) // 2] = self.priority_queue[child_index]
                child_index = 2*child_index +1
        self.priority_queue[(child_index-1) // 2] = x
        pass
    
    ## bottom-up adjusting queue
    def _build(self,):
        n = len(self.priority_queue)
        for i in range(n//2-1, -1, -1):
            self._adjust(i)
        pass
    
    ## pop item by index
    def _pop(self, i=0):
        '''

        :param stored_index:
        :param i: index of queue to pop, default the first item
        :param queue: [0,...,n-1], each store {constant.SIMILARITY: , constant.J_INDEX: }
        :type queue: [{}, {}]
        :param n: size of tree
        :return: queue[i] before pop
        '''
        n = len(self.priority_queue)
        if i >= n:
            raise IndexError("what you wanna pop is out of index")
        x = self.priority_queue[i]
        self.priority_queue[i] = self.priority_queue.pop()
        self._adjust(i,)
        return x
    
    ## delete item by item
    def delet_item(self, item:dict):
        self._pop(i=self.priority_queue.index(item))
        pass

    def insert(self, item:dict):
        ''' adaptive insert in heap

        :param stored_index:
        :param queue: [0,...,n-1], each store {constant.SIMILARITY: , constant.J_INDEX: }
        :type: [{}, {}]
        :param n: size of tree
        :type n: int
        :param item:
        :type: {constant.SIMILARITY:, constant.J_INDEX:}
        :return:
        '''
        self.priority_queue.append(item)
        n =len(self.priority_queue)
        child_index = n-1
        parent_index = (child_index-1)//2
        while 1:
            if self.priority_queue[parent_index][constant.SIMILARITY] >= item[constant.SIMILARITY]:
                break
            else:
                self.priority_queue[child_index] = self.priority_queue[parent_index]
                child_index = parent_index
                if child_index == 0:
                    break
                parent_index = (child_index-1)//2
        self.priority_queue[child_index] = item
        pass
    def clear(self):
        self.priority_queue = []
