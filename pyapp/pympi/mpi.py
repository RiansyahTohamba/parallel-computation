"""
MPI PARALLELISM COURSE FOR MACHINE LEARNING VECTOR OPERATIONS
==============================================================

Prerequisites:
- Install MPI: sudo apt-get install mpich (Linux) or brew install mpich (Mac)
- Install mpi4py: pip install mpi4py numpy

Run examples with: mpiexec -n 4 python this_file.py
"""

from mpi4py import MPI
import numpy as np
import time

# ============================================================================
# LESSON 1: MPI BASICS
# ============================================================================

def lesson1_hello_world():
    """Basic MPI initialization and process identification"""
    comm = MPI.COMM_WORLD  # Communicator object
    rank = comm.Get_rank()  # Process ID (0 to size-1)
    size = comm.Get_size()  # Total number of processes
    
    print(f"Hello from process {rank} of {size}")
    
    return comm, rank, size


# ============================================================================
# LESSON 2: POINT-TO-POINT COMMUNICATION
# ============================================================================

def lesson2_send_receive():
    """Send and receive data between processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        # Master sends data
        data = np.array([1.0, 2.0, 3.0, 4.0])
        print(f"Rank 0 sending: {data}")
        comm.send(data, dest=1, tag=11)
    elif rank == 1:
        # Worker receives data
        data = comm.recv(source=0, tag=11)
        print(f"Rank 1 received: {data}")


# ============================================================================
# LESSON 3: COLLECTIVE COMMUNICATION - SCATTER
# ============================================================================

def lesson3_scatter_vectors():
    """Distribute vector chunks to all processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        # Create a large vector on master
        vector = np.arange(16, dtype='float64')
        print(f"Master vector: {vector}")
        # Split into equal chunks
        chunks = np.array_split(vector, size)
    else:
        chunks = None
    
    # Scatter chunks to all processes
    local_data = comm.scatter(chunks, root=0)
    print(f"Rank {rank} received: {local_data}")
    
    return local_data


# ============================================================================
# LESSON 4: COLLECTIVE COMMUNICATION - GATHER
# ============================================================================

def lesson4_gather_results():
    """Collect results from all processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Each process has local data
    local_data = np.array([rank * 2, rank * 2 + 1], dtype='float64')
    print(f"Rank {rank} has: {local_data}")
    
    # Gather all data to master
    gathered = comm.gather(local_data, root=0)
    
    if rank == 0:
        # Flatten the list of arrays
        result = np.concatenate(gathered)
        print(f"Master gathered: {result}")
        return result


# ============================================================================
# LESSON 5: REDUCTION OPERATIONS
# ============================================================================

def lesson5_parallel_sum():
    """Compute sum across all processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Each process computes local sum
    local_array = np.array([rank + 1, rank + 2, rank + 3], dtype='float64')
    local_sum = np.sum(local_array)
    print(f"Rank {rank} local sum: {local_sum}")
    
    # Reduce all local sums to master
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Global sum: {total_sum}")
    
    return total_sum


# ============================================================================
# ML APPLICATION 1: PARALLEL DOT PRODUCT
# ============================================================================

def ml_app1_parallel_dot_product():
    """Compute dot product of two large vectors in parallel"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 1000000  # Vector size
    
    if rank == 0:
        # Create vectors on master
        v1 = np.random.randn(N)
        v2 = np.random.randn(N)
        
        # Time sequential version
        start = time.time()
        seq_result = np.dot(v1, v2)
        seq_time = time.time() - start
        
        # Split for parallel computation
        v1_chunks = np.array_split(v1, size)
        v2_chunks = np.array_split(v2, size)
    else:
        v1_chunks = None
        v2_chunks = None
        seq_result = None
        seq_time = None
    
    # Distribute chunks
    local_v1 = comm.scatter(v1_chunks, root=0)
    local_v2 = comm.scatter(v2_chunks, root=0)
    
    # Start parallel timing
    comm.Barrier()  # Synchronize
    start = time.time()
    
    # Compute local dot product
    local_dot = np.dot(local_v1, local_v2)
    
    # Sum all local results
    global_dot = comm.reduce(local_dot, op=MPI.SUM, root=0)
    
    comm.Barrier()
    par_time = time.time() - start
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("PARALLEL DOT PRODUCT RESULTS")
        print(f"{'='*60}")
        print(f"Vector size: {N}")
        print(f"Number of processes: {size}")
        print(f"Sequential time: {seq_time:.6f}s")
        print(f"Parallel time: {par_time:.6f}s")
        print(f"Speedup: {seq_time/par_time:.2f}x")
        print(f"Sequential result: {seq_result:.6f}")
        print(f"Parallel result: {global_dot:.6f}")
        print(f"Difference: {abs(seq_result - global_dot):.2e}")


# ============================================================================
# ML APPLICATION 2: PARALLEL MATRIX-VECTOR MULTIPLICATION
# ============================================================================

def ml_app2_matrix_vector_multiply():
    """Matrix-vector multiplication for neural network layer"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    rows, cols = 8000, 5000
    
    if rank == 0:
        # Create weight matrix and input vector
        W = np.random.randn(rows, cols) * 0.01
        x = np.random.randn(cols)
        
        # Sequential computation
        start = time.time()
        y_seq = W @ x
        seq_time = time.time() - start
        
        # Split matrix by rows
        W_chunks = np.array_split(W, size, axis=0)
    else:
        x = None
        W_chunks = None
        y_seq = None
        seq_time = None
    
    # Broadcast input vector to all processes
    x = comm.bcast(x, root=0)
    
    # Scatter matrix rows
    local_W = comm.scatter(W_chunks, root=0)
    
    # Parallel computation
    comm.Barrier()
    start = time.time()
    
    local_y = local_W @ x
    
    # Gather results
    y_par = comm.gather(local_y, root=0)
    
    comm.Barrier()
    par_time = time.time() - start
    
    if rank == 0:
        y_par = np.concatenate(y_par)
        print(f"\n{'='*60}")
        print("MATRIX-VECTOR MULTIPLICATION RESULTS")
        print(f"{'='*60}")
        print(f"Matrix shape: {rows}x{cols}")
        print(f"Sequential time: {seq_time:.6f}s")
        print(f"Parallel time: {par_time:.6f}s")
        print(f"Speedup: {seq_time/par_time:.2f}x")
        print(f"Max difference: {np.max(np.abs(y_seq - y_par)):.2e}")


# ============================================================================
# ML APPLICATION 3: PARALLEL GRADIENT COMPUTATION
# ============================================================================

def ml_app3_gradient_descent_step():
    """Parallel gradient computation for mini-batch SGD"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n_samples = 10000
    n_features = 1000
    
    if rank == 0:
        # Generate dataset
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        w = np.random.randn(n_features)
        
        # Split data
        X_chunks = np.array_split(X, size, axis=0)
        y_chunks = np.array_split(y, size)
    else:
        w = None
        X_chunks = None
        y_chunks = None
    
    # Broadcast weights
    w = comm.bcast(w, root=0)
    
    # Distribute data
    local_X = comm.scatter(X_chunks, root=0)
    local_y = comm.scatter(y_chunks, root=0)
    
    # Compute local gradient
    local_pred = local_X @ w
    local_error = local_pred - local_y
    local_grad = (local_X.T @ local_error) / len(local_y)
    
    # Average gradients across all processes
    global_grad = np.zeros_like(local_grad)
    comm.Reduce(local_grad, global_grad, op=MPI.SUM, root=0)
    
    if rank == 0:
        global_grad /= size
        print(f"\n{'='*60}")
        print("PARALLEL GRADIENT COMPUTATION")
        print(f"{'='*60}")
        print(f"Dataset: {n_samples} samples, {n_features} features")
        print(f"Processes: {size}")
        print(f"Gradient norm: {np.linalg.norm(global_grad):.6f}")
        
        # Update weights
        learning_rate = 0.01
        w_new = w - learning_rate * global_grad
        print(f"Weight update norm: {np.linalg.norm(w_new - w):.6f}")


# ============================================================================
# ML APPLICATION 4: PARALLEL K-MEANS CLUSTERING
# ============================================================================

def ml_app4_kmeans_iteration():
    """One iteration of parallel K-means"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n_samples = 10000
    n_features = 50
    k = 10
    
    if rank == 0:
        # Generate data
        X = np.random.randn(n_samples, n_features)
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        
        X_chunks = np.array_split(X, size, axis=0)
    else:
        centroids = None
        X_chunks = None
    
    # Broadcast centroids
    centroids = comm.bcast(centroids, root=0)
    
    # Distribute data
    local_X = comm.scatter(X_chunks, root=0)
    
    # Assign points to nearest centroid
    distances = np.zeros((len(local_X), k))
    for i in range(k):
        distances[:, i] = np.linalg.norm(local_X - centroids[i], axis=1)
    
    local_labels = np.argmin(distances, axis=1)
    
    # Compute local centroids
    local_sums = np.zeros((k, n_features))
    local_counts = np.zeros(k)
    
    for i in range(k):
        mask = local_labels == i
        local_counts[i] = np.sum(mask)
        if local_counts[i] > 0:
            local_sums[i] = np.sum(local_X[mask], axis=0)
    
    # Reduce to get global centroids
    global_sums = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)
    
    comm.Reduce(local_sums, global_sums, op=MPI.SUM, root=0)
    comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
    
    if rank == 0:
        new_centroids = global_sums / global_counts[:, np.newaxis]
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        
        print(f"\n{'='*60}")
        print("PARALLEL K-MEANS ITERATION")
        print(f"{'='*60}")
        print(f"Data: {n_samples} samples, {n_features} features")
        print(f"Clusters: {k}")
        print(f"Centroid shift: {centroid_shift:.6f}")
        print(f"Cluster sizes: {global_counts}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*60)
        print("MPI PARALLELISM COURSE FOR MACHINE LEARNING")
        print("="*60)
        print("\nUncomment sections below to run different lessons\n")
    
    # Uncomment to run specific lessons:
    
    # lesson1_hello_world()
    # lesson2_send_receive()
    # lesson3_scatter_vectors()
    # lesson4_gather_results()
    # lesson5_parallel_sum()
    
    # ML Applications (uncomment one at a time):
    ml_app1_parallel_dot_product()
    # ml_app2_matrix_vector_multiply()
    # ml_app3_gradient_descent_step()
    # ml_app4_kmeans_iteration()


"""
EXERCISES FOR PRACTICE:
=======================

1. Modify ml_app1 to compute cosine similarity instead of dot product

2. Implement parallel computation of vector norm (L2 norm)

3. Create a parallel version of ReLU activation: max(0, x)

4. Implement parallel computation of mean and standard deviation

5. Build a parallel logistic regression training loop

6. Create parallel mini-batch data loading

PERFORMANCE TIPS:
=================

1. Minimize communication - compute locally when possible
2. Use collective operations (scatter, gather, reduce) over point-to-point
3. Balance workload across processes
4. Use MPI.Barrier() sparingly - only for timing
5. Consider communication overhead vs computation time
6. Use numpy operations for vectorization

COMMON PITFALLS:
================

1. Not synchronizing before timing operations
2. Forgetting to initialize variables on all processes
3. Using send/recv when broadcast/scatter would work better
4. Not checking if rank == 0 before master-only operations
5. Unequal data distribution causing load imbalance
"""