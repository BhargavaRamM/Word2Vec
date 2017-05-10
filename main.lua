require ("io")
require ("os")
require ("paths")
require ("torch")
require ("word2vector")

config = {}
config.corpus = "corpus.txt" -- input data
config.window = 5
config.dim = 100
config.alpha = 0.75
config.table_size = 1e8
config.neg_samples = 5
config.minfreq = 10
config.lr = 0.025
config.min_lr = 0.001
config.epochs = 3
config.gpu = 1 -- 0 = use cpu and 1 = use gpu
config.stream = 1 -- 1 = stream from hard drive 0 = copy into memory

--parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window", config.window)
cmd:option("-minfreq", config.minfreq)
cmd:option("-dim", config.dim)
cmd:option("-lr", config.lr)
cmd:option("-min_lr", config.min_lr)
cmd:option("-neg_samples", config.neg_samples)
cmd:option("-table_size", config.table_size)
cmd:option("-epochs", config.epochs)
cmd:option("-gpu", config.gpu)
cmd:option("-stream", config.stream)
params = cmd:parse(arg)

for params, value in pairs(params) do
	config[params] = value
end

for i,j in pairs(config) do
	print(i..": "..j)
end

m = word2vec(config)
m:wordFrequency(config.corpus)
--m:build_table()
m:trimVocab()
m.lr = config.lr
m:wordTable()
m:buildModel()
for k=1, config.epochs do
	m.lr = config.min_lr
	m:trainModel(config.corpus)
end


--for k = 1, config.epochs do
--	m.lr = config.lr
--	m:train_model(config.corpus)
--end
m:print_sim_words({"the","he","can"}, 5)
