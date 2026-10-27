"""
案例：
    演示detach()函数功能，解决自动微分的弊端

回顾：
    自动微分 = 求导，即：基于损失函数，计算梯度
    结合权重更新公式：W新 = W旧 - 学习率 * 梯度，来更新权重

问题:
    一个张量一旦设置了自动微分，这个张量就不能直接转化为 numpy 的 ndnarray对象了，需要通过detach()函数解决
"""
# 导包
import torch
import numpy as np

# 1.定义张量
t1 = torch.tensor([10,20],requires_grad=True,dtype=torch.float)
print(f't1:{t1} , type:{type(t1)}')

# 2.尝试把上述张量转换成
# n1 = t1.numpy()      # 报错
# print(f'n1:{n1} , type:{type(n1)}')

# 3.解决方法：通过detach()函数，拷贝一份张量，然后转换
t2 = t1.detach()
print(f't2:{t2} , type:{type(t2)}')

# 4.测试上述t1和t2是否共享内存 ->  共享空间
t1.data[0] = 100    # 这个.data相当于去拿t1的数据
print(f't1:{t1} , type:{type(t1)}')
print(f't2:{t2} , type:{type(t2)}')
print('-' * 30)

# 5.查看t1和t2谁可以自动微分
print(f't1:{t1.requires_grad}')
print(f't2:{t2.requires_grad}')
print('-' * 30)

# 6.把t2转numpy对象
n1 = t2.numpy()
print(f'n1:{n1} , type:{type(n1)}')

# 7.最终版
n2 = t1.detach().numpy()
print(f'n2:{n2} , type:{type(n2)}')


