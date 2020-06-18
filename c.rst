**Numpy 切片和索引**
====================
Ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。

Ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。

**实例1**

In[1]: import numpy as np 

MyArray1 = np.arange(10)     # [0 1 2 3 4 5 6 7 8 9]

s = slice(2,7,2)     # 从索引 2 开始到索引 7 停止，间隔为2

print (MyArray1[s])

Out[1]: [2 4 6]

**实例2**

In[2]: import numpy as np   

MyArray1 = np.arange(10)     # [0 1 2 3 4 5 6 7 8 9]

MyArray1[1:9:2]     # 从索引 1 开始到索引 9 停止，间隔为 2


Out[2]:  array([1, 3, 5, 7])

**实例3**

In[3]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[:9:2]

Out[3]:  array([0, 2, 4, 6, 8])

**实例4**

In[4]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[::2]

Out[4]:  array([0, 2, 4, 6, 8])

**实例5**

In[5]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[::]

Out[5]:  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

**实例6**

In[6]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[:8:]

Out[6]:  array([0, 1, 2, 3, 4, 5, 6, 7])

**实例7**

In[7]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[:8]

Out[7]:  array([0, 1, 2, 3, 4, 5, 6, 7])

**实例8**

In[8]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[4::]

Out[8]:  array([4, 5, 6, 7, 8, 9])

**实例9**

In[9]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[9:1:-2]

Out[9]:  array([9, 7, 5, 3])

**实例10**

In[10]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[::-2]

Out[10]:  array([9, 7, 5, 3, 1])

**实例11**  布尔索引

In[11]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[MyArray1>5]

Out[11]:  array([6, 7, 8, 9])

**实例12**

In[12]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[5]

Out[12]:  5

**实例13**

In[13]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[-1]

Out[13]:  9

**实例14**  花式索引

In[14]: import numpy as np   

MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[[2,5,6]]

Out[14]:  array([2, 5, 6])

**实例15**  整数数组索引

In[15]: import numpy as np   

x = np.array([[1,  2],  [3,  4],  [5,  6]]) 

y = x[[0,1,2],  [0,1,0]]   #获取数组中(0,0)，(1,1)和(2,0)位置处的元素

y

Out[15]:  array([1, 4, 5])

**实例16**  

In[16]: import numpy as np   
 
MyArray1 = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]

MyArray1[:,np.newaxis]  #在指定位置增加一个维度

Out[16]:  array([[0],

       [1],

       [2],

       [3],

       [4],

       [5],

       [6],

       [7],

       [8],

       [9]])

**实例17**

In[17]: import numpy as np   

a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print(a[...,1] )  #...魔法糖

Out[17]:  [2 4 5]    
