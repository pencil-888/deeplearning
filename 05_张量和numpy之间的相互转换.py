"""
案例:
    演示张量和numpy之间如何转换，以及如何从张量中提取内容

涉及到的函数：
    场景1：张量 -> numpy nd数组
          张量对象.numpy()              共享内存
          张量对象.numpy().copy()       不共享内存
    场景2：numpy nd数组 -> 张量
          from_numpy()                共享内存
          torch.tensor(nd数组)不       不共享内存
    场景3：从标量中提取内容
          标量张量.item()

需要掌握的：
    张量 -> numpy：张量.numpy()
    numpy -> 张量：torch.tensor(nd数组)
    从标量中提取内容：标量张量,item()
"""

# 1.定义函数，演示：张量 -> numpy

