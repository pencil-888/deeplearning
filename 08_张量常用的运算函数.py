"""
案例：
    演示张量常用的运算函数
涉及到的函数：
    sum(), max(), min(), mean()          -> 都有dim参数，0表示列 1表示行
    pow(), sqrt(), exp(), log(), log2()  -> 都没有dim参数
需要掌握的函数：
    sum(), max(), min(), mean()
"""

# 导包
import torch

# 1.定义张量，记录初值
t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],],dtype=torch.float)
print(f't1:{t1}')

# 2.演示有dim参数的函数
# sum()求和
print(t1.sum(dim=0))   # 按列求和
print(t1.sum(dim=1))   # 按行求和
print(t1.sum())        # 整体求和
print('-'*30)

# max()求最大值，min()同理，这里暂不做演示
print(t1.max(dim=0))   # 按列求最大值
print(t1.max(dim=1))   # 按行求最大值
print(t1.max())        # 整体求最大值
print('-'*30)

# mean()求平均值  mean()必须是浮点型，要不然会报错
print(t1.mean(dim=0))   # 按列求平均值
print(t1.mean(dim=1))   # 按行求平均值
print(t1.mean())        # 整体求平均值
print('-'*30)

# 3.演示没有dim参数的函数
# pow() n次幂
print(t1.pow(2))
print(t1.pow(3))
print(t1 ** 2)
print(t1 ** 3)
print('-'*30)

# sqrt()平方根
print(t1.sqrt())
print('-'*30)

# exp() e的n次幂，n就是矩阵中的每个元素，这里是e^1,e^2,e^3,e^4,e^5,e^6
print(t1.exp())
print('-'*30)

# log(),log2(),log10()
print(t1.log())
print(t1.log2())
print(t1.log10())
