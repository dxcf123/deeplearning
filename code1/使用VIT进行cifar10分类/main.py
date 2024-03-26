import function

batchsize = 32
patchsize = 4
path = r'C:\Users\30535\Desktop\CodeProgram\Python\deepstudy\data'

model = function.train(path, batchsize, patchsize, device='cuda')
