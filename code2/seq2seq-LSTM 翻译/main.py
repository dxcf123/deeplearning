import function
import torch

path = r'./data/translate.csv'
path2 = r'./data'

# 返回模型，可以进行后处理
model = function.train(path, path2, lr=0.00001, epochs=30, batchsize=5, device='cuda')
torch.save(model.state_dict(), './data/model.pth')

