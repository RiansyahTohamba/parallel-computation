"""
Intel Iris Xe GPU Parallel Computing Tutorial with NLP
======================================================

This tutorial teaches GPU parallel computing using Intel's oneAPI and dpctl
for accelerating NLP tasks on Intel Iris Xe integrated GPUs.

Prerequisites:
- Intel CPU with Iris Xe integrated GPU
- Install: pip install dpctl dpnp numpy scikit-learn
- Install Intel oneAPI Base Toolkit (for full GPU support)
"""

import numpy as np
import time

# ============================================================================
# SECTION 1: Understanding CPU vs GPU Parallelism
# ============================================================================

def section_1_cpu_parallel():
    """
    CPU Parallel Processing: Good for complex branching logic
    - Few cores (4-16 typically)
    - Each core is powerful and independent
    - Best for: sequential tasks, complex control flow
    """
    print("=" * 70)
    print("SECTION 1: CPU Parallel Processing")
    print("=" * 70)
    
    # Example: Processing sentences sequentially
    sentences = [
        "Natural language processing is fascinating",
        "GPUs accelerate parallel computations",
        "Intel Iris Xe provides integrated graphics",
        "Python makes GPU programming accessible"
    ] * 250  # 1000 sentences
    
    start = time.time()
    
    # CPU operation: word counting
    word_counts = []
    for sentence in sentences:
        word_counts.append(len(sentence.split()))
    
    cpu_time = time.time() - start
    print(f"✓ Processed {len(sentences)} sentences on CPU")
    print(f"⏱ Time: {cpu_time:.4f} seconds")
    print(f"📊 Average words per sentence: {np.mean(word_counts):.2f}")
    
    return cpu_time

# ============================================================================
# SECTION 2: Intel Iris Xe GPU Architecture Basics
# ============================================================================

def section_2_gpu_architecture():
    print("=" * 70)
    print("SECTION 2: GPU CHECK (SAFE MODE)")
    print("=" * 70)
    try:
        import dpctl

        print("\n🚀 Enumerating SYCL devices...")

        try:
            devices = dpctl.get_devices()
        except Exception as e:
            print("❌ Tidak bisa mengambil daftar device.")
            print("   Penyebab:", e)
            raise SystemExit()

        if not devices:
            print("⚠ Tidak ada device SYCL yang terdeteksi.")
            print("   Kemungkinan penyebab:")
            print("   - Intel GPU driver belum ter-install (Level Zero / Compute Runtime).")
            print("   - oneAPI Base Toolkit belum ter-install.")
            raise SystemExit()

        for i, dev in enumerate(devices):
            print(f"\n[{i}] {dev.name}")
            print(f"    Backend     : {dev.backend}")
            print(f"    Type        : {dev.device_type}")
            print(f"    Compute Unit: {dev.max_compute_units}")

        # Coba ambil device GPU secara spesifik (jika ada)
        gpu_candidates = [d for d in devices if d.device_type == "gpu"]
        if gpu_candidates:
            print("\n✓ GPU ditemukan!")
            gpu = gpu_candidates[0]
            print("  GPU:", gpu.name)
        else:
            print("\n⚠ Tidak ada GPU yang terdeteksi oleh SYCL.")
            print("   (Iris Xe tidak muncul sebagai device SYCL)")

    except ImportError:
        print("❌ dpctl belum ter-install.")
        print("   Install dengan: pip install dpctl")

# ============================================================================
# SECTION 3: Simple GPU Operations - Vector Addition
# ============================================================================

def section_3_simple_gpu_operation(device):
    """
    First GPU Program: Vector addition
    This demonstrates basic data transfer and parallel execution
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Your First GPU Program - Vector Addition")
    print("=" * 70)
    
    try:
        import dpctl
        import dpnp
        
        # Create data
        size = 1_000_000
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        
        # CPU version
        start = time.time()
        c_cpu = a + b
        cpu_time = time.time() - start
        
        # GPU version
        with dpctl.device_context(device):
            # Transfer data to GPU
            a_gpu = dpnp.array(a)
            b_gpu = dpnp.array(b)
            
            start = time.time()
            c_gpu = a_gpu + b_gpu  # Executed on GPU
            c_gpu_result = dpnp.asnumpy(c_gpu)  # Transfer back to CPU
            gpu_time = time.time() - start
        
        # Verify correctness
        matches = np.allclose(c_cpu, c_gpu_result)
        
        print(f"✓ Vector size: {size:,} elements")
        print(f"⏱ CPU time: {cpu_time:.4f}s")
        print(f"⏱ GPU time: {gpu_time:.4f}s")
        print(f"🚀 Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"✓ Results match: {matches}")
        
    except ImportError as e:
        print(f"⚠ Error: {e}")
        print("   Install with: pip install dpnp")

# ============================================================================
# SECTION 4: NLP Case Study - Text Vectorization (TF-IDF)
# ============================================================================

def section_4_nlp_vectorization(device):
    """
    NLP Application: Parallel TF-IDF computation
    
    TF-IDF (Term Frequency-Inverse Document Frequency):
    - Measures word importance in documents
    - Highly parallelizable: each document processed independently
    """
    print("\n" + "=" * 70)
    print("SECTION 4: NLP Case Study - TF-IDF Vectorization")
    print("=" * 70)
    
    # Sample corpus
    documents = [
        "machine learning algorithms process data efficiently",
        "natural language processing uses neural networks",
        "deep learning models require large datasets",
        "gpu acceleration improves training speed",
        "parallel computing enables faster processing",
        "intel iris xe provides integrated graphics power",
        "python programming simplifies complex tasks",
        "artificial intelligence transforms modern technology"
    ] * 125  # 1000 documents
    
    print(f"📚 Corpus size: {len(documents)} documents")
    
    # Build vocabulary
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    vocab = sorted(list(vocab))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    print(f"📖 Vocabulary size: {len(vocab)} unique words")
    
    # CPU version
    start = time.time()
    cpu_tfidf = compute_tfidf_cpu(documents, vocab, word_to_idx)
    cpu_time = time.time() - start
    
    # GPU version (using dpnp for parallel operations)
    try:
        import dpnp
        
        with dpctl.device_context(device):
            start = time.time()
            gpu_tfidf = compute_tfidf_gpu(documents, vocab, word_to_idx, dpnp)
            gpu_time = time.time() - start
            
            print(f"\n⏱ CPU time: {cpu_time:.4f}s")
            print(f"⏱ GPU time: {gpu_time:.4f}s")
            print(f"🚀 Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Show sample results
            print(f"\n📊 Sample TF-IDF scores (first document):")
            top_words = np.argsort(cpu_tfidf[0])[-5:][::-1]
            for idx in top_words:
                print(f"   '{vocab[idx]}': {cpu_tfidf[0][idx]:.4f}")
    
    except ImportError:
        print("\n⚠ dpnp not available - showing CPU results only")
        print(f"⏱ CPU time: {cpu_time:.4f}s")

def compute_tfidf_cpu(documents, vocab, word_to_idx):
    """CPU implementation of TF-IDF"""
    n_docs = len(documents)
    n_vocab = len(vocab)
    
    # Term frequency matrix
    tf = np.zeros((n_docs, n_vocab), dtype=np.float32)
    
    for i, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            if word in word_to_idx:
                tf[i, word_to_idx[word]] += 1
        
        # Normalize by document length
        if len(words) > 0:
            tf[i] /= len(words)
    
    # Inverse document frequency
    df = np.sum(tf > 0, axis=0)
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    
    # TF-IDF
    tfidf = tf * idf
    
    return tfidf

def compute_tfidf_gpu(documents, vocab, word_to_idx, dpnp):
    """GPU-accelerated implementation of TF-IDF"""
    n_docs = len(documents)
    n_vocab = len(vocab)
    
    # Build TF matrix on CPU (tokenization is still CPU-bound)
    tf_cpu = np.zeros((n_docs, n_vocab), dtype=np.float32)
    
    for i, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            if word in word_to_idx:
                tf_cpu[i, word_to_idx[word]] += 1
        if len(words) > 0:
            tf_cpu[i] /= len(words)
    
    # Transfer to GPU for parallel operations
    tf = dpnp.array(tf_cpu)
    
    # Parallel IDF computation on GPU
    df = dpnp.sum(tf > 0, axis=0)
    idf = dpnp.log((n_docs + 1) / (df + 1)) + 1
    
    # Parallel TF-IDF multiplication
    tfidf = tf * idf
    
    # Transfer back to CPU
    return dpnp.asnumpy(tfidf)

# ============================================================================
# SECTION 5: Advanced - Semantic Similarity with GPU
# ============================================================================

def section_5_semantic_similarity(device):
    """
    Advanced NLP: Computing document similarities in parallel
    
    Cosine similarity between all document pairs - O(n²) operation
    Perfect for GPU parallelization!
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Semantic Similarity Computation")
    print("=" * 70)
    
    # Generate random document vectors (simulating embeddings)
    n_docs = 500
    embedding_dim = 128
    
    print(f"📊 Computing similarities for {n_docs} documents")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Total comparisons: {n_docs * n_docs:,}")
    
    # Random embeddings
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # CPU version
    start = time.time()
    similarities_cpu = np.dot(embeddings, embeddings.T)
    cpu_time = time.time() - start
    
    # GPU version
    try:
        import dpnp
        
        with dpctl.device_context(device):
            emb_gpu = dpnp.array(embeddings)
            
            start = time.time()
            similarities_gpu = dpnp.dot(emb_gpu, emb_gpu.T)
            similarities_gpu_result = dpnp.asnumpy(similarities_gpu)
            gpu_time = time.time() - start
        
        matches = np.allclose(similarities_cpu, similarities_gpu_result, atol=1e-5)
        
        print(f"\n⏱ CPU time: {cpu_time:.4f}s")
        print(f"⏱ GPU time: {gpu_time:.4f}s")
        print(f"🚀 Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"✓ Results match: {matches}")
        
        # Find most similar documents
        np.fill_diagonal(similarities_cpu, -1)
        most_similar = np.argmax(similarities_cpu[0])
        print(f"\n📝 Document 0 is most similar to Document {most_similar}")
        print(f"   Similarity score: {similarities_cpu[0, most_similar]:.4f}")
        
    except ImportError:
        print(f"\n⏱ CPU time: {cpu_time:.4f}s")
        print("⚠ GPU comparison not available")

# ============================================================================
# SECTION 6: Best Practices & Optimization Tips
# ============================================================================

def section_6_best_practices():
    """
    Key concepts for efficient GPU programming
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Best Practices for GPU Programming")
    print("=" * 70)
    
    tips = """
    🎯 When to Use GPU Acceleration:
    ✓ Large data arrays (>10,000 elements)
    ✓ Matrix operations (dot products, matrix multiplication)
    ✓ Element-wise operations on arrays
    ✓ Repetitive operations on independent data
    
    ⚠ When NOT to Use GPU:
    ✗ Small data (overhead > computation time)
    ✗ Complex branching logic
    ✗ Sequential operations with dependencies
    ✗ Frequent CPU-GPU data transfers
    
    💡 Optimization Tips:
    1. Minimize CPU-GPU data transfers (biggest bottleneck)
    2. Batch operations together on GPU
    3. Use appropriate data types (float32 often faster than float64)
    4. Keep data on GPU between operations when possible
    5. Profile your code to identify bottlenecks
    
    📚 Intel Iris Xe Specifics:
    - Shared memory with CPU (no discrete transfer needed)
    - Best for: medium-sized workloads (1K-10M elements)
    - Good for: inference, batch processing, data preprocessing
    - Consider for: training small models, real-time applications
    """
    
    print(tips)

# ============================================================================
# MAIN TUTORIAL RUNNER
# ============================================================================

def run_tutorial():
    """
    Run the complete tutorial
    """
    print("\n" + "=" * 70)
    print(" Intel Iris Xe GPU Parallel Computing Tutorial")
    print(" NLP Case Study: Accelerating Text Processing")
    print("=" * 70)
    
    # Section 1: CPU baseline
    section_1_cpu_parallel()
    
    # Section 2: GPU architecture
    device = section_2_gpu_architecture()
    
    if device is None:
        print("\n⚠ GPU device not available. Tutorial limited to concepts only.")
        section_6_best_practices()
        return
    
    # # Section 3: First GPU program
    # section_3_simple_gpu_operation(device)
    
    # # Section 4: NLP vectorization
    # section_4_nlp_vectorization(device)
    
    # # Section 5: Semantic similarity
    # section_5_semantic_similarity(device)
    
    # # Section 6: Best practices
    # section_6_best_practices()
    
    print("\n" + "=" * 70)
    print(" 🎓 Tutorial Complete!")
    print(" Next steps:")
    print("   - Experiment with different data sizes")
    print("   - Try your own NLP tasks")
    print("   - Explore Intel Extension for PyTorch")
    print("   - Check out Intel oneAPI documentation")
    print("=" * 70)

if __name__ == "__main__":
    run_tutorial()