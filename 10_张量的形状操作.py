"""
案例：
    演示张量的形状操作

涉及到的函数：
    reshape()          在不改变张量内容，顺序的前提下，对其形状做改变
    unsqueeze()        在指定的轴上增加(1)维度，等价于：升维
    squeeze()          删除所有为1的维度，等价于：降维
    transpose()        一次同时只能交换两个维度
    permute()          一次可以同时交换多个维度
    view()             只能修改连续的形状，存储顺序和张量显示顺序保持一致即连续，也就是经过transpose 和 permute以后就无法使用view操作了
    contiguous()       把不连续的张量 -> 连续的张量 ，即:基于张量中显示的顺序，修改内存中的存储顺序
    is_contiguous()    判断张量是否连续

需要掌握的函数：
    reshape(),unsqueeze(),permute(),view()
"""

# 导包
import torch

torch.manual_seed(24)
# 1.定义函数，演示：reshape()函数
def dm01():
    # 1.定义2行3列的张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}, row:{t1.shape[0]},columns:{t1.shape[1]},{t1.shape[-1]}')

    # 2.通过reshape()函数将t1 -> 3行2列，1行6列，6行1列
    # t2 = t1.reshape(3,2)
    # t2 = t1.reshape(1,6)
    t2 = t1.reshape(6,1)
    print(f't2:{t2},shape:{t2.shape}, row:{t2.shape[0]},columns:{t2.shape[1]},{t2.shape[-1]}')

    # 3.尝试通过reshape()函数，把t1 -> 2行5列的结果
    # t3 = t1.reshape(2,5)   # 会报错！！！必须保证元素个数不变
    # print(f't3:{t3}')

# 2.定义函数，演示：unsqueeze()函数,squeeze()函数
def dm02():
    # 1.定义2行3列的张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')      # (2,3)

    # 2.在0维上，添加1个维度
    t2 = t1.unsqueeze(0)
    print(f't2:{t2},shape:{t2.shape}')     # (1,2,3)

    # 3.在1维上，添加1个维度
    t3 = t1.unsqueeze(1)
    print(f't3:{t3},shape:{t3.shape}')     # (2,1,3)

    # 4.在2维上，添加1个维度
    t4 = t1.unsqueeze(2)
    print(f't4:{t4},shape:{t4.shape}')     # (2,3,1)

    # 5.在3维上，添加1个维度
    # t5 = t1.unsqueeze(3)        # 报错，越界了
    # print(f't5:{t5},shape:{t5.shape}')

    # 6.删除所有维度为1的维度
    t6 = torch.randint(1,10,size=(2,1,3,1,1))
    print(f't6:{t6},shape:{t6.shape}')     # (2,1,3,1,1)

    t7 = t6.squeeze()
    print(f't7:{t7},shape:{t7.shape}')     # (2,3)

# 3.定义函数，演示：transpose()函数,permute()函数
def dm03():
    # 1.定义张量
    t1 = torch.randint(1,10,size=(2,3,4))
    print(f't1:{t1},shape:{t1.shape}')
    print('-'*30)

    # 2.改变维度从 (2,3,4) -> (3,2,4)
    # t2 = t1.transpose(0,2)
    t2 = t1.transpose(0,-1)     # 效果同上
    print(f't1:{t1},shape:{t1.shape}')  # 说明transpose是改变新的并不是改变原来的
    print(f't2:{t2},shape:{t2.shape}')

    # 3.改变维度从 (2,3,4) -> (4,2,3)
    t3 = t1.permute(2,0,1)
    print(f't3:{t3},shape:{t3.shape}')

# 4.定义函数，演示：contiguous()函数,is_contiguous()函数
def dm04():
    # 1.定义张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')

    # 2.判断张量是否连续
    # print(t1.is_contiguous())     #True

    # 3.通过view()函数，修改上述张量形状  从(2,3) -> (3,2)
    t2 = t1.view(3,2)
    print(f't2:{t2},shape:{t2.shape}')
    print(t2.is_contiguous())      # True

    # 4.通过transpose()交换维度 -> 交换之后，不连续了
    t3 = t1.transpose(0,1)
    print(f't3:{t3},shape:{t3.shape}')
    print(t3.is_contiguous())

    # 5.尝试把t3张量从(3,2),通过view转换成(2,3)
    # t4 = t3.view(2,3)               # t3不连续，所以view无法转换，报错
    # print(f't4:{t4},shape:{t4.shape}')

    # 6.可以通过contiguous()函数，把t3张量 -> 连续张脸 -> 然后才可以通过view修改形状
    t5 = t3.contiguous().view(2,3)
    print(f't5:{t5},shape:{t5.shape}')


# 5.测试函数
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()


