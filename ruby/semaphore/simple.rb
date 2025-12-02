class Semaphore
  def initialize(count)
    @count = count
    @mutex = Mutex.new
    @condition = ConditionVariable.new
  end

  def acquire
    @mutex.synchronize do
      while @count <= 0
        @condition.wait(@mutex)
      end
      @count -= 1
    end
  end

  def release
    @mutex.synchronize do
      @count += 1
      @condition.signal
    end
  end
end

# Contoh penggunaan
semaphore = Semaphore.new(3)
threads = []

10.times do |i|
  threads << Thread.new do
    semaphore.acquire
    puts "Thread #{i} is running"
    sleep(2) # Simulasi kerja
    puts "Thread #{i} is done"
    semaphore.release
  end
end

threads.each(&:join)
