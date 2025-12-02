require_relative 'simple'
class VirtualClassroom
    def initialize(max_participants)
      @semaphore = Semaphore.new(max_participants)
    end
  
    def join_classroom(student_id)
      @semaphore.acquire
      begin
        puts "Student #{student_id} joined the classroom."
        sleep(2) # Simulasi durasi di dalam kelas
      ensure
        puts "Student #{student_id} left the classroom."
        @semaphore.release
      end
    end
end

# Simulasi
classroom = VirtualClassroom.new(3) # Maksimal 3 peserta

threads = []

6.times do |i|
  threads << Thread.new { classroom.join_classroom(i) }
end

threads.each(&:join)