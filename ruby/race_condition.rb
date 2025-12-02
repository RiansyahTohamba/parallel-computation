@stock = 1

def payment
  if @stock > 0
    @stock -= 1 
    p "final stock: #{@stock}"
  else
    p "error! msg='stock of item is zero'"
  end
end

threads = 100.times.map do 
  Thread.new do
    payment
  end
end


threads.each(&:join)

