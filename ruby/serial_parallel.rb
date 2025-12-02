require 'securerandom'
require 'csv'
require 'benchmark'
require 'thread'

# Fungsi untuk menghasilkan data order acak
def generate_random_order
  id = SecureRandom.uuid
  sku = "SKU#{rand(1000..9999)}"
  total_price = rand(10.0..100.0).round(2)
  created_at = Time.now
  { id: id, sku: sku, total_price: total_price, created_at: created_at }
end

# Fungsi untuk menuliskan data order ke CSV secara serial
def write_orders_serial(filename, total_orders)
  CSV.open(filename, "wb") do |csv|
    csv << ["id", "sku", "total_price", "created_at"]
    total_orders.times do
      order = generate_random_order
      csv << [order[:id], order[:sku], order[:total_price], order[:created_at]]
    end
  end
end

# Fungsi untuk menuliskan data order ke CSV di file terpisah untuk setiap thread
def write_orders_thread(filename, total_orders)
    CSV.open(filename, "wb") do |csv|
      csv << ["id", "sku", "total_price", "created_at"]
      total_orders.times do
        order = generate_random_order
        csv << [order[:id], order[:sku], order[:total_price], order[:created_at]]
      end
    end
end
  
  # Fungsi untuk menuliskan data order ke beberapa file CSV menggunakan beberapa thread
def write_orders_parallel(total_orders, thread_count)
    orders_per_thread = total_orders / thread_count
    threads = []
  
    thread_count.times do |i|
      filename = "orders_thread_#{i + 1}.csv"
      threads << Thread.new do
        write_orders_thread(filename, orders_per_thread)
      end
    end
  
    threads.each(&:join)
  end
  
  # Fungsi untuk menggabungkan semua file CSV menjadi satu file utama
  def combine_csv_files(output_filename, thread_count)
    CSV.open(output_filename, "wb") do |output_csv|
      output_csv << ["id", "sku", "total_price", "created_at"]
      
      thread_count.times do |i|
        filename = "orders_thread_#{i + 1}.csv"
        CSV.foreach(filename, headers: true) do |row|
          output_csv << row
        end
      end
    end
  end

# Fungsi untuk menghitung waktu eksekusi
def measure_time
  Benchmark.realtime { yield }
end

# Jumlah transaksi=1.000.000
total_orders = 1000000

# Ukur waktu eksekusi versi serial
serial_time = measure_time do
    output_filename = "orders_serial.csv"
    write_orders_serial(output_filename, total_orders)
end

# Ukur waktu eksekusi versi paralel
parallel_time = measure_time do
    thread_count = 1000
    output_filename = "orders_parallel.csv"
    write_orders_parallel(total_orders, thread_count)
    # combine_csv_files(output_filename, thread_count)
end

# p generate_random_order
# serial_time= 100
# parallel_time = 100

puts "Waktu eksekusi (Serial): #{serial_time.round(2)} detik"
puts "Waktu eksekusi (Paralel): #{parallel_time.round(2)} detik"
