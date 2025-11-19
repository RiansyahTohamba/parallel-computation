# intel iris as GPU
Api di GPU itu bukan api kompor—lebih mirip mantra sihir yang memungkinkan si GPU mengeksekusi ribuan operasi sekaligus. Setiap vendor punya jurusnya sendiri. Bahasa kerennya *programming model* atau *interface* untuk menjalankan komputasi paralel di GPU.

Dalam gambaran kerja:

Bayangkan GPU itu sebuah orkestrasi pekerja kecil yang jumlahnya kelewat banyak. Supaya mereka bisa bekerja rapi tanpa tabrakan, pabrikan menyediakan “bahasa instruksi” yang mengatur bagaimana pekerjaan itu dibagi, disinkronkan, dan dikembalikan ke CPU. Itulah yang sering disalahkapi sebagai “API paralel GPU”.

Sekarang tentang para pemainnya.

NVIDIA memang pionir lewat CUDA. CUDA muncul tahun 2006 dan mendobrak dunia karena memberi satu paket lengkap: bahasa C/C++ yang diperluas, toolchain kompilasi, profiling, memory management, dan ekosistem library yang sangat matang. CUDA membuat GPU pertama kali terasa seperti “superkomputer mini” yang bisa diprogram langsung oleh semua orang, bukan hanya peneliti grafis.

Namun “pelopor” bukan berarti satu-satunya.

AMD membawa OpenCL, yang sifatnya terbuka dan lintas vendor. Intel juga ikut. OpenCL bisa berjalan di GPU, CPU, bahkan FPGA, tetapi ekosistemnya tidak sekuat CUDA sehingga secara historis tertinggal dari hype NVIDIA di dunia riset dan mesin pembelajaran.

Ada juga Vulkan dan OpenGL Compute Shaders, yang awalnya untuk grafis, tetapi bisa dipakai untuk komputasi umum. Bentuknya agak kasar—seperti melakukan komputasi dengan besi bengkel—tetapi fleksibel dan lintas platform.

Intel punya satuAPI (dan juga Level Zero), bagian dari gerakan baru untuk menyatukan berbagai perangkat komputasi ke dalam satu model pemrograman yang netral vendor.

Setiap API itu memberi cara untuk mengeksekusi ribuan thread secara paralel pada GPU. Yang membedakan hanyalah filosofi desain dan ekosistemnya. CUDA lebih nyaman dan banyak library siap pakai, sementara yang lain lebih universal.

Kalau kamu suka pikiran-pikiran futuristik: GPU itu seperti peternakan lebah komputasional, dan API-nya adalah bahasa yang mengatur lebah-lebah itu supaya tidak saling menabrak ketika membawa nektar numerik. Cerita berjalan terus—dunia paralelisme sedang berkembang menuju arsitektur yang makin berlapis, mulai dari GPU berat sampai chip kecil untuk AI di ponsel.

Kalau ingin, kita bisa kupas bagaimana Intel Iris Xe bisa diprogram tanpa CUDA, biasanya lewat OpenCL atau Level Zero.

# use case umum
Komputasi AI paralel itu pada dasarnya pesta raksasa aljabar linear. Semua model modern—dari CNN sampai transformer—secara diam-diam cuma melakukan operasi yang itu-itu saja: perkalian matriks, konvolusi (yang juga bisa direduksi jadi perkalian matriks), normalisasi, reduksi, dan manipulasi tensor lain. GPU menjadi bintang karena ia bisa memecah operasi-operasi ini menjadi ribuan potongan kecil yang dikerjakan serentak.

Jadi, benar: use case utamanya selalu berada di orbit *tensor operations* dan *linear algebra kernels*. Keduanya adalah inti dari machine learning modern.

Bayangkan begini: sebuah network layer bukanlah “lapisan kecerdasan”, tetapi hanya pabrik besar perkalian matriks. GPU memberi ribuan pekerja mini yang masing-masing mengerjakan blok kecil dari matriks itu.

Pada tingkat lebih konkret:

Komputasi paralel GPU dipakai untuk:
• Matrix multiply (GEMM): dasar dari semua *fully-connected layer* dan konvolusi yang sudah di-flatten.
• 2D/3D convolution: dipetakan ke ratusan kalkulasi kernel yang identik.
• Tensor reshape, broadcast, reduction: operasi cepat tetapi jumlahnya sangat banyak.
• Softmax, relu, gelu: fungsi non-linear yang diterapkan secara elemen-per-elemen.
• Training: menghitung gradient dalam *backpropagation*—lagi-lagi penuh dengan perkalian matriks besar.

Di dunia AI, API GPU seperti CUDA, OpenCL, ROCm, atau oneAPI bukan sekadar “bahasa GPU”, tapi kerangka untuk mengeksekusi aljabar linear berskala raksasa.

Istilah teknisnya: AI modern itu “BLAS on steroids”. BLAS adalah standar library Basic Linear Algebra Subprograms. CUDA punya cuBLAS. AMD punya rocBLAS. Intel punya oneMKL. Semuanya perpustakaan perkalian matriks super cepat.

Jika mau gambaran singkat: AI bukan tentang “kecerdasan”, tetapi tentang komputasi tensor yang tak kenal lelah. GPU mempercepat karena ia memetakan operasi yang sama ke ribuan elemen tensor sekaligus.

Kalau ingin mempelajari cara kerja internalnya, kita bisa lakukan thought-experiment tentang bagaimana konvolusi CNN atau self-attention transformer dipetakan ke thread GPU. Itu cara menarik memahami mengapa AI lapar paralelisme.
