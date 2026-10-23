"""
案例：
    创建指令类型的张量

涉及到的函数：
    type(torch支持的数据类型)
    half()/double()/float()/short()/int()/long()

需要掌握的函数：
    type()
"""

# 导包
import torch

# 场景1：直接创建指定类型的张量
t1 = torch.tensor([1, 2, 3, 4, 5])  # 没写类型默认int64
print(f't1: {t1}, (t1中元素)类型：{t1.dtype}, (张量)类型：{type(t1)}')

t2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)  # 默认是float32
print(f't2: {t2}, (t2中元素)类型：{t2.dtype}, (张量)类型：{type(t2)}')
print('-' * 30)

# 场景2：创建好张量后 -> 做类型转换
# 思路1：type(torch支持的数据类型)
t3 = t1.type(torch.int16)
print(f't3: {t3}, (t3中元素)类型：{t3.dtype}, (张量)类型：{type(t3)}')
print('-' * 30)

# 思路2：half()/double()/float()/short()/int()/long()
print(t3.half())  # float16
print(t3.double())  # float32,默认
print(t3.float())  # float64
print(t3.short())  # int16
print(t3.int())  # int32
print(t3.long())  # int64,默认
