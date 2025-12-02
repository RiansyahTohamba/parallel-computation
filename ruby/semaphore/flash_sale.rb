require_relative 'simple'
class FlashSale
  def initialize(stock, max_concurrent_users)
    @stock = stock
    @semaphore = Semaphore.new(max_concurrent_users)
    @mutex = Mutex.new
  end

  def process_order(user_id)
    @semaphore.acquire
    begin
      @mutex.synchronize do
        if @stock > 0
          @stock -= 1
          puts "User #{user_id} successfully ordered. Remaining stock: #{@stock}"
        else
          puts "User #{user_id} failed to order. Out of stock."
        end
      end
    ensure
      @semaphore.release
    end
  end
end


# Simulasi
flash_sale = FlashSale.new(5, 2) # 5 stok, maksimal 2 pengguna bersamaan

threads = []

10.times do |i|
  threads << Thread.new { flash_sale.process_order(i) }
end

threads.each(&:join)
