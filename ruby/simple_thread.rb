# Definisikan sebuah array untuk menyimpan threads
threads = []

# Membuat 5 thread
30.times do |i|
  threads << Thread.new do
    puts "Thread #{i} sedang berjalan"
    sleep(rand(1..3))  # Simulasi proses dengan sleep antara 1 hingga 3 detik
    puts "Thread #{i} selesai"
  end
end

# Tunggu hingga semua threads selesai
threads.each(&:join)

puts "Semua thread selesai dijalankan"
