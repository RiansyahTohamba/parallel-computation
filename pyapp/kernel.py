import dpctl
import numpy as np
from numba_dpex import kernel, DEFAULT_LOCAL_SIZE

# Kernel sederhana: menjumlahkan dua array
@kernel
def add_vectors(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]

# Pilih device Intel GPU
device = dpctl.SyclDevice("level_zero:gpu")

n = 100000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.zeros_like(a)

# Jalankan kernel di GPU Intel
with dpctl.SyclQueue(device) as q:
    add_vectors[n, DEFAULT_LOCAL_SIZE](a, b, c)

print("Hasil elemen pertama:", c[0])
print("Device yang dipakai:", device)
