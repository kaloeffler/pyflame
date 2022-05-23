import cffi
from pathlib import Path

print("Building CFFI Module")

current_dir = Path().absolute()
header_file = current_dir / "flame.h"

ffi = cffi.FFI()

ffi.cdef(
    """typedef struct Flame Flame;
typedef struct IntArray IntArray;
typedef float (*DistFunction)( float *x, float *y, int m );
struct Flame
{
	int simtype;

	/* Number of objects */
	int N;

	/* Number of K-Nearest Neighbors */
	int K;

	/* Upper bound for K defined as: sqrt(N)+10 */
	int KMAX;

	/* Stores the KMAX nearest neighbors instead of K nearest neighbors
	 * for each objects, so that when K is changed, weights and CSOs can be
	 * re-computed without referring to the original data.
	 */
	int   **graph;
	/* Distances to the KMAX nearest neighbors. */
	float **dists;

	/* Nearest neighbor count.
	 * it can be different from K if an object has nearest neighbors with
	 * equal distance. */
	int    *nncounts;
	float **weights;

	/* Number of identified Cluster Supporting Objects */
	int cso_count;
	char *obtypes;

	float **fuzzyships;
	
	/* Number of clusters including the outlier group */
	int count;
	/* The last one is the outlier group. */
	IntArray *clusters;
	
	DistFunction distfunc;
};
Flame* Flame_New(void);
Flame *Flame_Clustering(Flame *flame, float *data[], int N, int M, int knn, float thd, int steps, float epsilon);"""
)

ffi.set_source(
    "__pyflame",
    # Since you're calling a fully-built library directly, no custom source
    # is necessary. You need to include the .h files, though, because behind
    # the scenes cffi generates a .c file that contains a Python-friendly
    # wrapper around each of the functions.
    """
    # include "flame.h"
    """,
    # The important thing is to include the pre-built lib in the list of
    # libraries you're linking against:
    sources=["flame.c"],
    libraries=["m"],
    library_dirs=[current_dir.as_posix()],
    extra_link_args=["-Wl,-rpath,."],
)

ffi.compile(verbose=True)
