mutex = Mutex.new
condition = ConditionVariable.new
queue = []

producer = Thread.new do
  mutex.synchronize do
    puts "Producing item"
    queue << "Item from queue producer"
    condition.signal # Memberi sinyal bahwa item sudah ditambahkan
  end
end

consumer = Thread.new do
  mutex.synchronize do
    puts "Consumer waiting"
    condition.wait(mutex) if queue.empty? # Menunggu sampai ada item dalam queue
    puts "Consuming #{queue.pop}"
  end
end

producer.join
consumer.join
