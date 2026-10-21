"""
案例：
    演示如何创建全0，全1，指定值的张量

涉及到的函数：
    torch.ones 和 torch.ones_like创建全1的张量
    torch.zeros 和 torch.zeros_like创建全0的张量
    torch.full 和 torch.full_like创建全为指定值的1张量

需要掌握的函数;
    zeros(),full()
"""

# 导包
import torch

# 场景1：torch.ones 和 torch.ones_like创建全1的张量
def dm01():
    t1 = torch.ones(2, 3)  # 创建2行3列全1的张量
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 30)

# t2：3行2列
    t2 = torch.tensor([[1, 2], [3, 4],[5, 6]])
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 30)

# t3 -> 基于t2的形状，创建全1的张量
    t3 = torch.ones_like(t2)
    print(f't3 = {t3}, type = {type(t3)}')    #3行2列 -> 全1的矩阵
    print('-' * 30)



# 场景2：torch.zeros 和 torch.zeros_like创建全0的张量
def dm02():
    t1 = torch.zeros(2, 3)  # 创建2行3列全0的张量
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 30)

# t2：3行2列
    t2 = torch.tensor([[1, 2], [3, 4],[5, 6]])
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 30)

# t3 -> 基于t2的形状，创建全0的张量
    t3 = torch.zeros_like(t2)
    print(f't3 = {t3}, type = {type(t3)}')    #3行2列 -> 全1的矩阵
    print('-' * 30)



# 场景3：torch.full 和 torch.full_like创建全为指定值的1张量
def dm03():
    t1 = torch.full(size=(2,3), fill_value = 255)  # 创建2行3列全255的张量
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 30)

# t2：3行2列
    t2 = torch.tensor([[1, 2], [3, 4],[5, 6]])
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 30)

# t3 -> 基于t2的形状，创建全255的张量
    t3 = torch.full_like(t2,fill_value = 255)
    print(f't3 = {t3}, type = {type(t3)}')    #3行2列 -> 全255的矩阵
    print('-' * 30)

# 测试函数
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()