# readme

Intel Iris Xe itu GPU terpadu yang bisa dipakai di Python, tetapi tidak lewat CUDA. Jalan yang paling masuk akal adalah memakai ekosistem Intel oneAPI—khususnya numba-dpex (dulunya numba-sycl) atau dpctl. Ini memberi Python akses ke GPU lewat standar SYCL.

Di bawah ini contoh paling sederhana yang biasanya berhasil di laptop HP dengan Iris Xe, selama oneAPI toolkit dan paket Python pendukungnya sudah terpasang.

Kodenya tidak bergaya heroik—hanya menunjukkan bahwa komputasi benar-benar dijalankan di GPU.