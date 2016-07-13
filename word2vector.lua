require "sys"
require "nn"
require "torch"
total_count = 0
vocab = {}
word = torch.IntTensor(1)
termIndex = {}
terms = {}
vocab_size = 0

function word2vec:wordFrequency(corpus)
	-- Get the word frequency in the corpus. 
	local file = io.open(corpus,"r")
	for line in file:lines() do
		for _,word in ipairs(self.seperateLine(line)) do
			total_count = total_count+1
			if self.vocab[word] == nil then
				self.vocab[word] = 1
			else
				self.vocab[word] = self.vocab[word]+1
			end
		end
	end
	file.close()
end

function trimVocab( vocab,minfreq)
	for word,count in pairs(self.vocab) do
		if count >= minfreq then
			self.termIndex[#self.termIndex+1] = word
			self.terms[word] = #self.termIndex
		else
			self.vocab[word] = nil
		end
	end

	self.vocab_size = #self.termIndex
	io.write(string.format("Number of words in vocabulary after trimming with minimum frequency of %d is %d",self.minfreq, #self.vocab_size))
end

function seperateLine(inputLine)
	local words={}
	local i=1
	for word in string.gmatch(inputLine, "([^".."%s".."]+)") do 
		words[i] = word
		i=i+1
	end
	return words
end

function neural_model()
	self.wordVectors = nn.LookupTable(self.vocab_size,self.dimensions)
	self.contextVectors = nn.LookupTable(self.vocab_size,self.dimensions)
	self.wordVectors:reset(0.2)
	self.contextVectors:reset(0.2)
	self.model = nn.Sequential()
	self.model:add(nn.ParallelTable())
	self.model.modules[1]:add(self.wordVectors)
	self.model.modules[1]:add(self.contextVectors)
	self.model:add(nn.MM(false,true))
	self.model:add(Sigmoid())
end

function decay_rate(min_lr, learning_rate,window)
	decay = (self.min_lr-self.learning_rate)/(self.total_count*self.window)
end

