import pickle
import embedding
import cluster

import os
from os import listdir
from os.path import isfile, join, isdir

from tqdm import tqdm

from efficientnet_pytorch import EfficientNet

def load_image_data(data_path:str, inner_path:str="", top_k:int=5) -> dict:
    model = EfficientNet.from_pretrained("efficientnet-b0")
    dirs = [data_path + "/" + x + inner_path for x in os.listdir(data_path) if isdir(data_path + "/" + x + inner_path)]
    data_dict = {}
    for dir in tqdm(dirs):
        image_files = [join(dir, f) for f in listdir(dir)][:top_k]
        features = []
        for file in image_files:
            try:
                features.append(embedding.extract_features(file, model=model))
            except:
                continue

        data_dict[dir.split("/")[-1 if inner_path=="" else -2]] = features

    return data_dict

def save_data(data_dict:dict, filename:str="data.emb"):
    with open(filename, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename:str="data.emb"):
    with open(filename, "rb") as handle:
        return pickle.load(handle)

if __name__ == "__main__":
    data = load_image_data("../data/train", inner_path="/images")
    save_data(data)


