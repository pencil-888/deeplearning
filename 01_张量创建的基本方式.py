"""
常用快捷键：
    1.CTRL D直接复制行
    2.ALT SHIFT选择多列
    3.CTRL X剪切
    4.CTRL SHIFT L调整代码格式
    5.CTRL R replace 批量替换
    6.ALT 4 显示右侧的运行输出结果   ALT 1 显示左侧的工具栏（需要按两次哦）
    7.CTRL ALT INS 按上下键找到py新建py文件
    8.光标放到报错的地方ALT 回车 选择导入名称 就可以自动导包
    9.ALT SHIFT 移动代码行
案例：
    演示张量的创建方式

张量(Tensor)：
    PyTorch框架属于最常用的深度学习框架，无论是后续学习的ANN（人工神经网络）,还是CNN（卷积神经网络）,RNN（循环神经网络）
    底层在处理数据时，都是使用张量来处理的

    张量 -> 存储同一数据类型的容器，且元素值必须是数值才行

张量的基本创建方式：
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量，其也可以用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

细节：
    Tensor可以基于形状直接创建张量。
"""

# 导包
import torch
import numpy as np  # 导入 numpy，并给它起简称 np


# 1.定义函数，演示：torch.tensor 根据指定数据创建张量
def dm01():
    # 场景1：标量张量
    t1 = torch.tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    # 场景2：二维列表 -> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)

    # 场景3.1：numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))  # .random：访问 NumPy 中的随机数模块  .randint：调用生成随机整数的函数
    t3 = torch.tensor(data)  # 默认整型
    print(f't3:{t3},type:{type(t3)}')

    # 场景3.2：
    data = np.random.randint(0, 10, size=(2, 3))  # .random：访问 NumPy 中的随机数模块  .randint：调用生成随机整数的函数
    t3 = torch.tensor(data, dtype=torch.float)  # 改变了类型
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)

    # 场景4:尝试直接创建指定维度（例如：2行3列）的张量
    # t4 = torch.tensor(2,3)
    # print(f't4:{t4},type:{type(t4)}')    # 报错


# 2.定义函数，演示：torch.Tensor 根据形状创建张量，其也可以用来创建指定数据的张量
def dm02():
    # 场景1：标量张量
    t1 = torch.Tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    # 场景2：二维列表 -> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)

    # 场景3：numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))  # .random：访问 NumPy 中的随机数模块  .randint：调用生成随机整数的函数
    t3 = torch.Tensor(data)  # 你上面创建是啥就是啥，现在不能指定类型
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)

    # 场景4:尝试直接创建指定维度（例如：2行3列）的张量
    t4 = torch.Tensor(2, 3)
    print(f't4:{t4},type:{type(t4)}')


# 3.定义函数，演示：torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
def dm03():
    # 场景1：标量张量
    t1 = torch.IntTensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    # 场景2：二维列表 -> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)

    # 场景3：numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))  # .random：访问 NumPy 中的随机数模块  .randint：调用生成随机整数的函数
    t3 = torch.IntTensor(data)  # 你上面创建是啥就是啥，现在不能指定类型
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)

    # 场景4：如果类型不匹配，会尝试自动转换类型
    data = np.random.randint(0, 10, size=(2, 3))  # .random：访问 NumPy 中的随机数模块  .randint：调用生成随机整数的函数
    t4 = torch.FloatTensor(data)
    print(f't4:{t4},type:{type(t4)}')  # 默认float32，所以在输出时不显示
    print('-' * 30)


# 4.定义测试函数
if __name__ == '__main__':
    dm01()  # 掌握！
    # dm02()
    # dm03()
