from typing import List, Tuple
from dataloader import load_data
from collections import defaultdict

class TreeNode:
    def __init__(self, class_ptrs:List[int], data_ptrs:List[int]): 
        self.class_ptrs = class_ptrs
        self.data_ptrs = data_ptrs
        self.children = []

    def subdivide(self, labels, thresh=0.9) -> bool:
        count_of_labels = defaultdict(int)
        for class_ptr in self.class_ptrs:
            count_of_labels[labels[class_ptr]] += 1

        max_count = max(count_of_labels.items(), key=lambda x:x[1])
        return max_count/len(self.class_ptrs) < thresh

        
            

def perform_csl(x:List[List[float]], y:List[str], thresh:float) -> TreeNode:
    root = TreeNode(list(range(len(y))), list(range(len(y))))
    Q = [root]
    
    while len(Q) != 0:
        current_node = Q.pop(0)
        if current_node.subdivide(y, thresh=thresh):
            continue
            

