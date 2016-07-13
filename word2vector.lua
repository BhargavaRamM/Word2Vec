require "sys"
require "nn"
require "torch"
total_count = 0
vocab = {}
word = torch.IntTensor(1)
termIndex = {}
terms = {}
vocab_size = 0
alpha = 0
table_size = config.table_size

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
-- index2word is termIndex
-- word2index is terms
-- word is word
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
	self.mlp = nn.Sequential()
	self.mlp:add(nn.ParallelTable())
	self.mlp.modules[1]:add(self.wordVectors)
	self.mlp.modules[1]:add(self.contextVectors)
	self.mlp:add(nn.MM(false,true))
	self.mlp:add(Sigmoid())
end

function decay_rate(min_lr, learning_rate,window)
	decay = (self.min_lr-self.learning_rate)/(self.total_count*self.window)
end

function word_table() 
	local start_time = sys.clock()
	local total_count = 0
	for _,count in pairs(self.vocab) do
		total_count = total_count + count^alpha
	end
	self.table = torch.IntTensor(self.table_size)
	local word_idx = 1
	local word_prob = self.vocab[self.termIndex[word_idx]]^self.alpha/total_count
	for i = 1, self.table_size do
		self.table[i] = word_idx
		if i/self.table_size > word_prob then
			word_idx = word_idx + 1
			word_prob = word_prob + self.vocab[self.termIndex[word_idx]]^alpha/total_count
		end
		if word_idx > self.vocab_size then 
			word_idx = word_idx -1
		end
	end

	print(string.format("A word table is built in %.2f secs", sys.clock()-start_time))
end
-- lr is the learning_rate
-- mlp is w2v 
-- x is p and bp is dl_dp
function  train(word, contexts) 
	local x = self.mlp:forward({contexts,word})
	local loss = self.criterion:forward(x,self.labels)
	local bp = self.criterion:backward(x,self.labels)
	self.mlp:zeroGradParameters()
	self.mlp:backward({contexts,word},bp)
	self.mlp:updateParameters(self.learning_rate)
end

function sample_contexts(context) 
	self.contexts[1] = context
	local  i = 0
	while i< self.neg_samples do
		neg_context = self.table[torch.random(self.table_size)]
		if context ~= neg_context then
			self.contexts[i+2] = neg_context
			i = i+1
		end
	end
end

function train_corpus(corpus) 
	local start_time = sys.clock()
	local count = 0
	fileName = io.open(corpus,"r")
	for line in fileName:lines() do
		sentence = self.seperateLine(line)
		for i, word in ipairs(sentence) do
			word_index = self.terms[word]
			if word_index ~= nil then
				local win = torch.random(self.window)
				self.word[1] = word_index
				for j = i-win, i+win do
					local context = sentence[j]
					if context ~=nil and j~=i then
						context_index = self.terms[context]
						if context_index ~= nil then
							self.sample_contexts(context_index)
							self.train(self.word,self.contexts)
							count = count+1
							local decay = self.decay_rate(self.min_lr,learning_rate,window) 
							self.learning_rate = math.max(self.min_lr,self,learning_rate+decay)
							if count%100000 == 0 then 
								print(string.format("%d words trained in %.2f secs",count, sys.clock()-start_time))
							end
						end
					end
				end
			end
		end
	end
	print(string.format("%d words processed in %0.2f secs", count, sys.clock()-start_time))
end

function normalize(m)
	normalized_m = torch.zeros(m:size())
	for i=1,m:size(1) do
		normalized_m[i] = m[i]/torch.norm(m[i])
	end
	return normalized_m
end

function cuda()
	local cunn = require "cunn"
	local cutorch = require "cutorch"
	cutorch.setDevice(1)
	self.word = self.word:cuda()
	self.contexts = self.contexts:cuda()
	self.labels = self.labels:cuda()
	self.criterion:cuda()
	self.mlp:cuda()
end

function similar_words(w,k)
	if self.normalized_wordVectors == nil then
		self.normalized_wordVectors = self.normalize(self.wordVectors.weight:double())
	end
	if type(w) == "string" then
		if self.terms[w] == nil then
			print("Word doesn't exist in vocabulary")
			return nil
		else
			w = self.normalized_wordVectors[self.terms[w]]

		end
	end
	local similarity = torch.mv(self.normalized_wordVectors,w)
	similarity, idx = torch.sort(-similarity)
	local res = {}
	for i = 1, k do
		res[i] = {self.termIndex[idx[i]], -similarity[i]}
	end
	return res
end

function print_similar_words(words,k)
	for i = 1, #words do
		res = similar_words(words[i],k)
		if res ~= nil then
			print("----"..words[i].."----")
			for j = 1,k do
				print(string.format("%s, %.4f", res[j][1],res[j][2]))
			end
		end
	end
end

