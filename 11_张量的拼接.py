"""
案例：
    演示张量的拼接操作

涉及到的API:
    cat()          不会改变维度数，拼接张量，除了拼接的那个维度外，其他维度数必须保持一致
    stack()        会改变维度数，拼接张量，所有维度都必须保持一直
"""

# 导包
import torch
torch.manual_seed(24)
# 1.创建两个张量
t1 = torch.randint(1,10,(2,3))
print(f't1:{t1},shape:{t1.shape}')

t2 = torch.randint(1,10,(2,3))
print(f't2:{t2},shape:{t2.shape}')
print('-'*30)

# 2.演示张量的拼接
# 思路1：cat()拼接张量
t3 = torch.cat([t1,t2],dim=0)       # (2,3) + (2,3) = (4,3)
print(f't3:{t3},shape:{t3.shape}')         # 由于这个地方是0维度拼接，故0维度可以不一样，其他维度必须保持一样，可以是(2,3) + (4,3) = (6,3)

# t4 = torch.cat([t1,t2],dim=1)       # (2,3) + (2,3) = (2,6)
# print(f't4:{t4},shape:{t4.shape}')         # 拼接的维度可以不同，其他维度必须保持相同！！！

# t5 = torch.cat([t1,t2],dim=-1)       # 效果同上
# print(f't5:{t5},shape:{t5.shape}')

# t6 = torch.cat([t1,t2],dim=2)       # 报错
# print(f't6:{t6},shape:{t6.shape}')

print('-'*30)

# 思路2：stack()拼接张量，可以是新维度，但是无论新旧维度，所有维度都必须保持一致，也就是张量必须是一摸一样的形状，(2,3)和(2,3)这样类似形状一致才可以拼接，
t7 = torch.stack([t1,t2], dim=0)    # (2,3) + (2,3) = (2,2,3)
print(f't7:{t7},shape:{t7.shape}')

t8 = torch.stack([t1,t2],dim=1)     # (2,3) + (2,3) = (2,2,3)
print(f't8:{t8},shape:{t8.shape}')

# 上面两个拼接虽然都显示是(2,2,3)但是拼接的维度不同哦~

t9 = torch.stack([t1,t2],dim=2)     # (2,3) + (2,3) = (2,3,2)
print(f't9:{t9},shape:{t9.shape}')

# 通过上述例子可以看出两个张量的stack拼接都是相对对应维度进行拼接


