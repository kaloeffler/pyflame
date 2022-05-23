import cffi
from pathlib import Path
print("Building CFFI Module")

current_dir = Path().absolute()
header_file = current_dir / "flame.h"

ffi = cffi.FFI()

ffi.cdef("int* Flame_Clustering(float *data[], int labels[], int N, int M, int knn, float thd, int steps, float epsilon, float cluster_assign_thd);")

ffi.set_source(
    "pyflame",
    # Since you're calling a fully-built library directly, no custom source
    # is necessary. You need to include the .h files, though, because behind
    # the scenes cffi generates a .c file that contains a Python-friendly
    # wrapper around each of the functions.
    """
    # include "flame.h"
    """,
    # The important thing is to include the pre-built lib in the list of
    # libraries you're linking against:
    sources = ["flame.c"],
    libraries=["m"],
    library_dirs=[current_dir.as_posix()],
    extra_link_args=["-Wl,-rpath,."],
)

ffi.compile(verbose=True)

