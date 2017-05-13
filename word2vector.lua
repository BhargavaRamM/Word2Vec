require "sys"
require "nn"
require "torch"

local Word2Vec = torch.class("word2vec")

--total_count = 0
--vocab = {}
--word = torch.IntTensor(1)
--termIndex = {}
--terms = {}
vocab_size = 0
--alpha = 0
--table_size = config.table_size
wordVectors = {}
contextVectors = {}


function word2vec:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.minfreq = config.minfreq
    self.dim = config.dim
    self.criterion = nn.BCECriterion() -- logistic loss
    self.word = torch.IntTensor(1) 
    self.contexts = torch.IntTensor(1+self.neg_samples) 
    self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 -- first label is always pos sample
    self.window = config.window 
    self.lr = config.lr 
    self.min_lr = config.min_lr
    self.alpha = config.alpha
    self.table_size = config.table_size 
    self.vocab = {}
    self.termIndex = {}
    self.terms = {}
    self.total_count = 0
end

function word2vec:wordFrequency(corpus)
	-- Get the word frequency in the corpus. 
	-- Build vocab
	local f = io.open(corpus,"r")

	for line in f:lines() do
		for _,word in ipairs(self:seperateLine(line)) do
			self.total_count = self.total_count + 1
			if self.vocab[word] == nil then
				self.vocab[word] = 1
			else
				self.vocab[word] = self.vocab[word]+1
			end
		end
	end
	f.close()
--	for key,val in pairs(self.vocab) do
--		print(key..":"..val)
--	end
end

-- index2word is termIndex
-- word2index is terms
-- word is word
function word2vec:trimVocab()
	for word,count in pairs(self.vocab) do
		if count >= self.minfreq then
			self.termIndex[#self.termIndex+1] = word
			self.terms[word] = #self.termIndex
		else
			self.vocab[word] = nil
		end
	end

	self.vocab_size = #self.termIndex
	io.write(string.format("Number of words in vocabulary after trimming with minimum frequency of %d is %d\n",self.minfreq, self.vocab_size))
end

function word2vec:seperateLine(inputLine)
	local words={}
	local i=1
	for word in string.gmatch(inputLine, "([^".."%s".."]+)") do 
		words[i] = word
		i=i+1
	end
	return words
end

function word2vec:buildModel()
    print (self.vocab_size.." << "..self.dim)
	self.wordVectors = nn.LookupTable(self.vocab_size,self.dim)
	self.contextVectors = nn.LookupTable(self.vocab_size,self.dim)
	self.wordVectors:reset(0.2)
	self.contextVectors:reset(0.2)
	self.mlp = nn.Sequential()
	self.mlp:add(nn.ParallelTable())
	self.mlp.modules[1]:add(self.wordVectors)
	self.mlp.modules[1]:add(self.contextVectors)
	self.mlp:add(nn.MM(false,true))
	self.mlp:add(nn.Sigmoid())
    print ("Model built")
end

function word2vec:decay_rate(min_lr, learning_rate,window)
    local decay = 0
	decay = (self.min_lr-self.lr)/(self.total_count*self.window)
    return decay
end

function word2vec:getTotalWeightOfWordProbs()
    local total_count = 0
    for _, count in pairs(self.vocab) do
--        print (total_count.." -- "..count.."--"..self.alpha)
        total_count = total_count + count^self.alpha
    end
    return total_count
end

function word2vec:wordTable()
	local start_time = sys.clock()
    total_count = self:getTotalWeightOfWordProbs()
	print("Total Count: "..total_count)
	self.table = torch.IntTensor(self.table_size)
	local word_idx = 1
--  insert each word index into table
	local word_prob = self.vocab[self.termIndex[word_idx]]^self.alpha/total_count
	for i = 1, self.table_size do
		self.table[i] = word_idx
		if i/self.table_size > word_prob then
			word_idx = word_idx + 1
			word_prob = word_prob + self.vocab[self.termIndex[word_idx]]^self.alpha/total_count
--            print (word_idx.." -- "..word_prob)
		end
		if word_idx > self.vocab_size then
			word_idx = word_idx - 1
		end
    end
	print(string.format("A word table is built in %.2f secs", sys.clock()-start_time))
end

-- lr is the learning_rate
-- mlp is w2v 
-- x is p and bp is dl_dp
function  word2vec:train(word, contexts)
    if self.gpu == 1 then
        self:cuda()
    end
	local x = self.mlp:forward({contexts,word})
	local loss = self.criterion:forward(x,self.labels)
	local bp = self.criterion:backward(x,self.labels)
	self.mlp:zeroGradParameters()
	self.mlp:backward({contexts,word},bp)
	self.mlp:updateParameters(self.lr)
end

function word2vec:sample_contexts(context) 
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

function word2vec:trainModel(corpus)
    local start_time = sys.clock()
    local count = 0
    fileName = io.open(corpus,"r")
    for line in fileName:lines() do
--        print (line)
        sentence = self:seperateLine(line)
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
                            self:sample_contexts(context_index)
                            self:train(self.word,self.contexts)
                            count = count+1
                            local decay = self:decay_rate(self.min_lr,self.lr,win)
                            self.lr = math.max(self.min_lr,self.lr+decay)
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
    torch.save("model.t7",self.mlp)
end

function word2vec:normalize(m)
	normalized_m = torch.zeros(m:size())
	for i=1,m:size(1) do
		normalized_m[i] = m[i]/torch.norm(m[i])
	end
	return normalized_m
end

function word2vec:cuda()
	local cunn = require "cunn"
	local cutorch = require "cutorch"
	cutorch.setDevice(1)
	self.word = self.word:cuda()
	self.contexts = self.contexts:cuda()
	self.labels = self.labels:cuda()
	self.criterion:cuda()
	self.mlp:cuda()
end

function word2vec:similar_words(w,k)
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

function word2vec:print_similar_words(words,k)
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


