import __pyflame as pyfl
import numpy as np

def flame_clustering(data: np.array, knn: int = 10, thd: float = -2.0, steps: int = 500, epsilon: float = 1e-6, fuzzy_clusters: bool = True, min_membership_thr: float = 0.5):
    """Python bindings to the FLAME clustering algorithm implemented in C.
    The orignial C code was implemented by Fu Limin and is available at https://github.com/zjroth/flame-clustering

    Args:
        latent_vectors (np.array): array of shape (N, M) where each row m represents a single vector and N is the dimension of the embedding
        knn (int, optional): number of nearest neighbors to consider for each sample in data. Defaults to 10.
        thd (float, optional): threshold. Defaults to -2.0.
        steps (int, optional): Number of iterations to perform. Defaults to 500.
        epsilon (float, optional): _description_. Defaults to 1e-6.
        fuzzy_clusters (bool, optional): If true return an array of shape [N;K] where each row represents the membership of the sample to each of the K clusters. Defaults to True. Otherwise return a [N] dim vector representing the id of the cluster
        with the highest score.
        min_membership_thr (float, optional): if fuzzy_clusters is False samples with a max membership score below this threshold will be assigned to the outlier class (-1). Defaults to 0.5
    """
    # allocates the specified C type and returns a pointer to it
    n,m = data.shape
    # allocates the specified C type and returns a pointer to it
    flame = pyfl.ffi.new(f"Flame*")
    flame = pyfl.lib.Flame_New()

    d_pointers = pyfl.ffi.new(f"float* [{n}]")
    # keep memory alive -> therefore the arrays need to be saved in a variable explicity!!!
    rows = [pyfl.ffi.new(f"float [{m}]") for i in range(n)]
    for i in range(n):
        d_pointers[i] = rows[i]
        for j in range(m):
            d_pointers[i][j] = float(data[i, j])

    pp = pyfl.ffi.cast("float **", d_pointers)
    flame = pyfl.lib.Flame_Clustering(flame, pp, n, m, knn, thd, steps, epsilon)


    # unpack the data to [N, K] where K is the number of clusters
    fuzzy_labels = [pyfl.ffi.unpack(elem, flame.cso_count) for elem in pyfl.ffi.unpack(flame.fuzzyships, N)]
    fuzzy_labels = np.stack(fuzzy_labels)

    if not fuzzy_clusters:
        labels = np.argmax(fuzzy_labels, axis=1)
        max_membership_score = np.max(fuzzy_labels, axis=1)
        labels[max_membership_score < min_membership_thr] = -1
        return labels
    return fuzzy_labels


if __name__ == "__main__":
    knn = 10
    thd = -2.0
    steps = 500
    epsilon = 1e-6
    cluster_thr = -1.0

    with open("matrix.txt", "r") as file:
        N, M = [int(elem) for elem in file.readline().lstrip("\n").split(" ")]
        data = np.zeros((N, M))

        for i, line in enumerate(file.readlines()):
            row = [float(elem) for elem in line.lstrip("\n").split(" ")]
            data[i,:] = row
    # fuzzy labels
    fuzzy_labels = flame_clustering(data, knn, thd, steps, epsilon, fuzzy_clusters=True, min_membership_thr=None)
    print("Fuzzy labels:")
    print(fuzzy_labels[:10])
    # discrete labels
    labels = flame_clustering(data, knn, thd, steps, epsilon, fuzzy_clusters=False, min_membership_thr=0.5)
    print("Discrete labels:")
    print(labels[:10])