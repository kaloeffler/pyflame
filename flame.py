import pyflame
import numpy as np


knn = 10
thd = -2.0
steps = 500
epsilon = 1e-6
cluster_thr = -1.0

with open("matrix.txt", "r") as file:
    N, M = [int(elem) for elem in file.readline().lstrip("\n").split(" ")]
    np_array = np.zeros((N, M))

    for i, line in enumerate(file.readlines()):
        row = [float(elem) for elem in line.lstrip("\n").split(" ")]
        np_array[i,:] = row


# allocates the specified C type and returns a pointer to it
labels = pyflame.ffi.new(f"int [{N}]")

d_pointers = pyflame.ffi.new(f"float* [{N}]")
# keep memory alive -> therefore the arrays need to be saved in a variable explicity!!!
rows = [pyflame.ffi.new(f"float [{M}]") for i in range(N)]
for i in range(N):
    d_pointers[i] = rows[i]
    for j in range(M):
        d_pointers[i][j] = float(np_array[i, j])

pp = pyflame.ffi.cast("float **", d_pointers)

print(pyflame.ffi.unpack(labels, N))
labels_edit = pyflame.lib.Flame_Clustering(pp, labels, N, M, knn, thd, steps, epsilon, cluster_thr)

np_labels = pyflame.ffi.unpack(labels_edit, N)

#pyflame.ffi.free(labels)
#pyflame.ffi.free(d_pointers)
print(np_labels)
