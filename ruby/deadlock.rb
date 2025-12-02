mutex1 = Mutex.new
mutex2 = Mutex.new

thread1 = Thread.new do
  mutex1.lock
  sleep 1 # Simulasikan pekerjaan
  mutex2.lock
  puts "Thread 1 selesai"
  mutex2.unlock
  mutex1.unlock
end

thread2 = Thread.new do
  mutex2.lock
  sleep 1 # Simulasikan pekerjaan
  mutex1.lock
  puts "Thread 2 selesai"
  mutex1.unlock
  mutex2.unlock
end

thread1.join
thread2.join
