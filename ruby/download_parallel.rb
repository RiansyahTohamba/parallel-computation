require 'net/http'
require 'json'

urls = [
  'https://jsonplaceholder.typicode.com/posts',
  'https://jsonplaceholder.typicode.com/comments',
  'https://jsonplaceholder.typicode.com/albums'
]

responses = {}

threads = urls.map do |url|
  Thread.new do
    uri = URI(url)
    response = Net::HTTP.get(uri)
    puts "Data from #{url} received"
    responses[url] = JSON.parse(response)
  end
end

threads.each(&:join)

File.open("jsonplaceholder.json", "w") do |file|
    file.write(JSON.pretty_generate(responses))
end

puts "All data downloaded"
