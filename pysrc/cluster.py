import umap
import hdbscan
import numpy as np

def dim_reduction(data_pts:list[list], init="spectral") -> list[list]:
    embeddings = umap.UMAP(n_neighbors=15,
                           n_components=128,
                           init=init,
                           metric='correlation').fit_transform(data_pts)
    return embeddings


def cluster(embeddings:list[list]) -> list[int]:
    clusterer = hdbscan.HDBSCAN()
    cluster_labels = clusterer.fit(embeddings)
    return cluster_labels.labels_

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    from efficientnet_pytorch import EfficientNet
    from tqdm import tqdm
    import embedding

    path = "../data/test/images"
    model = EfficientNet.from_pretrained("efficientnet-b0")

    print("loading images")
    image_files = [join(path, f) for f in listdir(path) if isfile(join(path, f))][:10]
    print("extracting features")

    features = []
    for file in tqdm(image_files):
        features.append(embedding.extract_features(file, model=model))

    print("finished extracting")
    embeddings = dim_reduction(features, init="random")
    print(np.array(embeddings).shape)

    clusters = cluster(embeddings)

    print(clusters.labels_)
