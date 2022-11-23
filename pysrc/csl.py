import numpy as np
from typing import List, Tuple
from dataloader import load_data
from collections import defaultdict
import cluster

class TreeNode:
    def __init__(self, class_ptrs:List[int], data_ptrs:List[int]): 
        self.class_ptrs = class_ptrs
        self.data_ptrs = data_ptrs
        self.children = []

    def subdivide(self, labels, thresh=0.9) -> bool:
        count_of_labels = defaultdict(int)
        for class_ptr in self.class_ptrs:
            count_of_labels[labels[class_ptr]] += 1

        max_count = max(count_of_labels.items(), key=lambda x:x[1])[1]
        return max_count/len(self.class_ptrs) < thresh

        
    def partition(self, data:List[List[np.array]]) -> Tuple[List[List], List[List]]:
        data_clusters = defaultdict(list)
        label_clusters = defaultdict(list)

        data_arr = [data[idx] for idx in self.data_ptrs]
        cluster_labels = cluster.cluster(data_arr)

        data_to_remove = []
        class_to_remove = []
        for idx, label in enumerate(cluster_labels):
            data_to_remove.append(self.data_ptrs[idx])
            class_to_remove.append(self.class_ptrs[idx])
            
            data_clusters[label].append(self.data_ptrs[idx])
            label_clusters[label].append(self.class_ptrs[idx])

        for data_idx, class_idx in zip(data_to_remove, class_to_remove):
            self.data_ptrs.remove(data_idx)
            self.class_ptrs.remove(class_idx)

        return (data_clusters, label_clusters)
            
    def __repr__(self) -> str:
        return f"Treenode: {len(self.data_ptrs)} {len(self.class_ptrs)}"

    def __str__(self) -> str:
        return self.__repr__()

def perform_csl(x:List[List[float]], y:List[str], thresh:float) -> TreeNode:
    root = TreeNode(list(range(len(y))), list(range(len(y))))
    Q = [root]
    
    while len(Q) != 0:
        print(len(Q))
        current_node = Q.pop(0);
        print(current_node)
        if current_node.subdivide(y, thresh=thresh):
            data_clusters, label_clusters = current_node.partition(x)
            for cluster_idx, data_ids in data_clusters.items():
                new_node = TreeNode(label_clusters[cluster_idx], data_ids)
                current_node.children.append(new_node)
                Q.append(new_node)

    return root
                

if __name__ == "__main__":
    from tqdm import tqdm

    data = load_data()
    x = []
    y = []
    for id, data_pts in tqdm(data.items()):
        x.extend(data_pts)
        y.extend([id] * len(data_pts))

    x = cluster.dim_reduction(x)

    print(len(x), len(y))

    perform_csl(x, y, 0.9)