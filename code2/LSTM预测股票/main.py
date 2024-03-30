import function

# 数据路径
path = './data/我大A.csv'

# 对接下来第几个工作日进行预测
late = 5
# 输入维度
input_dim = 1
# 隐藏层
hidden_dim = 64
# LSTM层数
num_layer = 2
# 输出维度
output_dim = 1
# epoch
epochs = 500
# batch_size
batch_size = 64

model = function.train(path=path, late=late, input_dim=input_dim, hidden_dim=hidden_dim, num_layer=num_layer,
                       output_dim=output_dim, epochs=epochs, batch_size=batch_size)
