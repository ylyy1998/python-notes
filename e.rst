**Numpy 数组操作**
=====================
**1、修改数组形状**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**实例1** 

numpy.reshape 函数可以在不改变数据的条件下修改形状，格式如下：

numpy.reshape(arr, newshape, order='C')

In[1]: import numpy as np
 
a = np.arange(8)

print ('原始数组：')

print (a)

print ('\n')
 
b = a.reshape(4,2)

print ('修改后的数组：')

print (b)

Out[1]:  原始数组：

[0 1 2 3 4 5 6 7]


修改后的数组：

[[0 1]

 [2 3]

 [4 5]

 [6 7]]

**实例2** 

numpy.ndarray.flat 是一个数组元素迭代器

In[2]: import numpy as np

a = np.arange(9).reshape(3,3)

print ('原始数组：')

for row in a:

    print (row)


#对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：

print ('迭代后的数组：')

for element in a.flat:
    
    print (element)

Out[2]:  原始数组：

[0 1 2]

[3 4 5]

[6 7 8]

迭代后的数组：

0

1

2

3

4

5

6

7

8

**实例3** 

numpy.ndarray.flatten 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组

ndarray.flatten(order='C')

In[3]: import numpy as np

a = np.arange(8).reshape(2,4)
 
print ('原数组：')

print (a)

print ('\n')

# 默认按行
 
print ('展开的数组：')

print (a.flatten())

print ('\n')
 
print ('以 F 风格顺序展开的数组：')

print (a.flatten(order = 'F'))

Out[3]: 原数组：

[[0 1 2 3]

 [4 5 6 7]]


展开的数组：

[0 1 2 3 4 5 6 7]


以 F 风格顺序展开的数组：

[0 4 1 5 2 6 3 7]

**2、翻转数组**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**实例1** 

numpy.transpose 函数用于对换数组的维度

numpy.transpose(arr, axes)

In[1]: import numpy as np

a = np.arange(12).reshape(3,4)
 
print ('原数组：')

print (a )

print ('\n')
 
print ('对换数组：')

print (np.transpose(a))

Out[1]:  原数组：

[[ 0  1  2  3]

 [ 4  5  6  7]

 [ 8  9 10 11]]


对换数组：

[[ 0  4  8]

 [ 1  5  9]

 [ 2  6 10]

 [ 3  7 11]]

**实例2** 

numpy.ndarray.T 类似 numpy.transpose

In[2]: import numpy as np

a = np.arange(12).reshape(3,4)
 
print ('原数组：')

print (a)

print ('\n')
 
print ('转置数组：')

print (a.T)

Out[2]: 原数组：

[[ 0  1  2  3]

 [ 4  5  6  7]

 [ 8  9 10 11]]


转置数组：

[[ 0  4  8]

 [ 1  5  9]

 [ 2  6 10]

 [ 3  7 11]]

**实例3** 

numpy.swapaxes 函数用于交换数组的两个轴

numpy.swapaxes(arr, axis1, axis2)

In[3]: import numpy as np

# 创建了三维的 ndarray

a = np.arange(8).reshape(2,2,2)
 
print ('原数组：')

print (a)

print ('\n')

# 现在交换轴 0（深度方向）到轴 2（宽度方向）
 
print ('调用 swapaxes 函数后的数组：')

print (np.swapaxes(a, 2, 0))

Out[3]: 原数组：

[[[0 1]

  [2 3]]

 [[4 5]

  [6 7]]]


调用 swapaxes 函数后的数组：

[[[0 4]

  [2 6]]

 [[1 5]

  [3 7]]]

**3、修改数组维度**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**实例1** 

numpy.broadcast 用于模仿广播的对象，它返回一个对象，该对象封装了将一个数组广播到另一个数组的结果。

In[1]: import numpy as np

x = np.array([[1], [2], [3]])

y = np.array([4, 5, 6])  
 
# 对 y 广播 x

b = np.broadcast(x,y)  

# 它拥有 iterator 属性，基于自身组件的迭代器元组
 
print ('对 y 广播 x：')

r,c = b.iters
 
# Python3.x 为 next(context) ，Python2.x 为 context.next()

print (next(r), next(c))

print (next(r), next(c))

print ('\n')

# shape 属性返回广播对象的形状
 
print ('广播对象的形状：')

print (b.shape)

print ('\n')

# 手动使用 broadcast 将 x 与 y 相加

b = np.broadcast(x,y)

c = np.empty(b.shape)
 
print ('手动使用 broadcast 将 x 与 y 相加：')

print (c.shape)

print ('\n')

c.flat = [u + v for (u,v) in b]
 
print ('调用 flat 函数：')

print (c)

print ('\n')

# 获得了和 NumPy 内建的广播支持相同的结果
 
print ('x 与 y 的和：')

print (x + y)

Out[1]: 对 y 广播 x：

1 4

1 5


广播对象的形状：

(3, 3)


手动使用 broadcast 将 x 与 y 相加：

(3, 3)


调用 flat 函数：

[[5. 6. 7.]

 [6. 7. 8.]

 [7. 8. 9.]]


x 与 y 的和：

[[5 6 7]

 [6 7 8]

 [7 8 9]] 

**实例2** 

numpy.broadcast_to 函数将数组广播到新形状。它在原始数组上返回只读视图。 它通常不连续。 如果新形状不符合 NumPy 的广播规则，该函数可能会抛出ValueError。

numpy.broadcast_to(array, shape, subok)

In[2]: import numpy as np

a = np.arange(4).reshape(1,4)
 
print ('原数组：')

print (a)

print ('\n')
 
print ('调用 broadcast_to 函数之后：')

print (np.broadcast_to(a,(4,4)))

Out[2]:  原数组：

[[0 1 2 3]]


调用 broadcast_to 函数之后：

[[0 1 2 3]

 [0 1 2 3]

 [0 1 2 3]

 [0 1 2 3]]

**4、连接数组**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**实例1** 

numpy.concatenate 函数用于沿指定轴连接相同形状的两个或多个数组，格式如下：

numpy.concatenate((a1, a2, ...), axis)

In[1]: import numpy as np

a = np.array([[1,2],[3,4]])
 
print ('第一个数组：')

print (a)

print ('\n')

b = np.array([[5,6],[7,8]])
 
print ('第二个数组：')

print (b)

print ('\n')

# 两个数组的维度相同
 
print ('沿轴 0 连接两个数组：')

print (np.concatenate((a,b)))

print ('\n')
 
print ('沿轴 1 连接两个数组：')

print (np.concatenate((a,b),axis = 1))

Out[1]: 第一个数组：

[[1 2]

 [3 4]]


第二个数组：

[[5 6]

 [7 8]]


沿轴 0 连接两个数组：

[[1 2]

 [3 4]

 [5 6]

 [7 8]]


沿轴 1 连接两个数组：

[[1 2 5 6]

 [3 4 7 8]]

**实例2** 

numpy.hstack 是 numpy.stack 函数的变体，它通过水平堆叠来生成数组。

In[2]: import numpy as np

a = np.array([[1,2],[3,4]])
 
print ('第一个数组：')

print (a)

print ('\n')

b = np.array([[5,6],[7,8]])
 
print ('第二个数组：')

print (b)

print ('\n')
 
print ('水平堆叠：')

c = np.hstack((a,b))

print (c)

print ('\n')

Out[2]:  第一个数组：

[[1 2]

 [3 4]]


第二个数组：

[[5 6]

 [7 8]]


水平堆叠：

[[1 2 5 6]

 [3 4 7 8]]

**实例3** 

numpy.vstack 是 numpy.stack 函数的变体，它通过垂直堆叠来生成数组。

In[3]: import numpy as np

a = np.array([[1,2],[3,4]])
 
print ('第一个数组：')

print (a)

print ('\n')

b = np.array([[5,6],[7,8]])
 
print ('第二个数组：')

print (b)

print ('\n')
 
print ('竖直堆叠：')

c = np.vstack((a,b))

print (c)

Out[3]:  第一个数组：

[[1 2]

 [3 4]]


第二个数组：

[[5 6]

 [7 8]]


竖直堆叠：

[[1 2]

 [3 4]

 [5 6]

 [7 8]]

**5、分割数组**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**实例1** 

numpy.split 函数沿特定的轴将数组分割为子数组，格式如下：

numpy.split(ary, indices_or_sections, axis)

In[1]: import numpy as np

a = np.arange(9)
 
print ('第一个数组：')

print (a)

print ('\n')
 
print ('将数组分为三个大小相等的子数组：')

b = np.split(a,3)

print (b)

print ('\n')
 
print ('将数组在一维数组中表明的位置分割：')

b = np.split(a,[4,7])

print (b)

Out[1]:  第一个数组：

[0 1 2 3 4 5 6 7 8]


将数组分为三个大小相等的子数组：

[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]


将数组在一维数组中表明的位置分割：

[array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8])]

**实例2** 

numpy.hsplit 函数用于水平分割数组，通过指定要返回的相同形状的数组数量来拆分原数组。

In[2]: import numpy as np

harr = np.floor(10 * np.random.random((2, 6)))

print ('原array：')

print(harr)
 
print ('拆分后：')

print(np.hsplit(harr, 3))

Out[2]: 原array：

[[2. 4. 7. 7. 8. 9.]

 [4. 4. 0. 1. 5. 8.]]

拆分后：

[array([[2., 4.],

       [4., 4.]]), array([[7., 7.],

       [0., 1.]]), array([[8., 9.],

       [5., 8.]])] 

**实例3** 

numpy.vsplit 沿着垂直轴分割，其分割方式与hsplit用法相同。

In[3]: import numpy as np

a = np.arange(16).reshape(4,4)
 
print ('第一个数组：')

print (a)

print ('\n')
 
print ('竖直分割：')

b = np.vsplit(a,2)

print (b)

Out[3]:  第一个数组：

[[ 0  1  2  3]

 [ 4  5  6  7]

 [ 8  9 10 11]

 [12 13 14 15]]


竖直分割：

[array([[0, 1, 2, 3],

       [4, 5, 6, 7]]), array([[ 8,  9, 10, 11],

       [12, 13, 14, 15]])]

**6、数组元素的添加与删除**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**实例1** 

numpy.resize 函数返回指定大小的新数组。如果新数组大小大于原始大小，则包含原始数组中的元素的副本。

numpy.resize(arr, shape)

In[1]: import numpy as np

a = np.array([[1,2,3],[4,5,6]])
 
print ('第一个数组：')

print (a)

print ('\n')
 
print ('第一个数组的形状：')

print (a.shape)

print ('\n')

b = np.resize(a, (3,2))
 
print ('第二个数组：')

print (b)

print ('\n')
 
print ('第二个数组的形状：')

print (b.shape)

print ('\n')

# 要注意 a 的第一行在 b 中重复出现，因为尺寸变大了
 
print ('修改第二个数组的大小：')

b = np.resize(a,(3,3))

print (b)

Out[1]:

第一个数组：

[[1 2 3]

 [4 5 6]]


第一个数组的形状：

(2, 3)


第二个数组：

[[1 2]

 [3 4]

 [5 6]]


第二个数组的形状：

(3, 2)


修改第二个数组的大小：

[[1 2 3]

 [4 5 6]

 [1 2 3]]

**实例2** 

numpy.append 函数在数组的末尾添加值。 追加操作会分配整个数组，并把原来的数组复制到新数组中。 此外，输入数组的维度必须匹配否则将生成ValueError。append 函数返回的始终是一个一维数组。

numpy.append(arr, values, axis=None)

In[2]: import numpy as np

a = np.array([[1,2,3],[4,5,6]])
 
print ('第一个数组：')

print (a)

print ('\n')
 
print ('向数组添加元素：')

print (np.append(a, [7,8,9]))

print ('\n')
 
print ('沿轴 0 添加元素：')

print (np.append(a, [[7,8,9]],axis = 0))

print ('\n')
 
print ('沿轴 1 添加元素：')

print (np.append(a, [[5,5,5],[7,8,9]],axis = 1))

Out[2]:  第一个数组：

[[1 2 3]

 [4 5 6]]


向数组添加元素：

[1 2 3 4 5 6 7 8 9]


沿轴 0 添加元素：

[[1 2 3]

 [4 5 6]

 [7 8 9]]


沿轴 1 添加元素：

[[1 2 3 5 5 5]

 [4 5 6 7 8 9]]

**实例3** 

numpy.delete 函数返回从输入数组中删除指定子数组的新数组。 与 insert() 函数的情况一样，如果未提供轴参数，则输入数组将展开。

Numpy.delete(arr, obj, axis)

In[3]: import numpy as np

a = np.arange(12).reshape(3,4)
 
print ('第一个数组：')

print (a)

print ('\n')
 
print ('未传递 Axis 参数。 在插入之前输入数组会被展开。')

print (np.delete(a,5))

print ('\n')
 
print ('删除第二列：')

print (np.delete(a,1,axis = 1))

print ('\n')
 
print ('包含从数组中删除的替代值的切片：')

a = np.array([1,2,3,4,5,6,7,8,9,10])

print (np.delete(a, np.s_[::2]))

Out[3]:  第一个数组：

[[ 0  1  2  3]

 [ 4  5  6  7]

 [ 8  9 10 11]]


未传递 Axis 参数。 在插入之前输入数组会被展开。

[ 0  1  2  3  4  6  7  8  9 10 11]


删除第二列：

[[ 0  2  3]

 [ 4  6  7]

 [ 8 10 11]]


包含从数组中删除的替代值的切片：

[ 2  4  6  8 10]