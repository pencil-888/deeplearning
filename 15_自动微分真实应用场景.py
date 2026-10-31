"""
案例：
    演示自动微分真实应用场景

结论：
    1.先正向传播计算出预测值z
    2.基于损失函数，结合预测值z和真实值y，来计算精度
    3.结合权重更新公式 W新 = W旧 - 学习率 * 梯度，来更新权重
"""

# 导包
import torch

# 1.定义x，表示特征（输入数据），假设：2行5列，全1矩阵
x = torch.ones(2,5)
print(x)

# 2.定义y，表示真实值，假设：2行3列，全0矩阵
y = torch.zeros(2,3)
print(y)

# 3.初始化（可自动微分的）权重和偏置
w = torch.randn(5,3,requires_grad=True)
print(w)

b = torch.randn(3,requires_grad=True)    # 由于是5行3列的w，故需要3个b来对应
print(b)

# 4.前向传播，计算出预测值z
z = torch.matmul(x,w) + b
# z = x @ w + b
print(z)

# 5.定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z,y)
print(loss)

# 6.进行自动微分，求导反向传播，更新权重
loss.backward()      # 如果loss不止一个值需要写loss.sum().backward()

# 7.打w，b用来更新的梯度
print(f'w的梯度={w.grad}')
print(f'b的梯度={b.grad}')

# 后续就是跟新权重
w.data = w.data - 0.01 * w.grad
print(w.data)







