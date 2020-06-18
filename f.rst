**Numpy函数**
=====================
**1、字符串函数**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
以下函数用于对 dtype 为 numpy.string_ 或 numpy.unicode_ 的数组执行向量化字符串操作。 它们基于 Python 内置库中的标准字符串函数。

**实例1**

numpy.char.add() 函数依次对两个数组的元素进行字符串连接。

In[1]: import numpy as np 

print ('连接两个字符串：')

print (np.char.add(['hello'],[' xyz']))

print ('\n')
 
print ('连接示例：')

print (np.char.add(['hello', 'hi'],[' abc', ' xyz']))

Out[1]: 连接两个字符串：

['hello xyz']


连接示例：

['hello abc' 'hi xyz']

**实例2**

numpy.char.multiply() 函数执行多重连接。

In[2]: import numpy as np 

print (np.char.multiply('Runoob ',3))

Out[2]: Runoob Runoob Runoob 

**实例3**

numpy.char.center() 函数用于将字符串居中，并使用指定字符在左侧和右侧进行填充。

In[3]: import numpy as np 

# np.char.center(str , width,fillchar) ：

# str: 字符串，width: 长度，fillchar: 填充字符

print (np.char.center('Runoob', 20,fillchar = '*'))

Out[3]: *******Runoob*******

**实例4**

numpy.char.capitalize() 函数将字符串的第一个字母转换为大写：

In[4]: import numpy as np 

print (np.char.capitalize('runoob'))

Out[4]: Runoob

**实例5**

numpy.char.title() 函数将字符串的每个单词的第一个字母转换为大写：

In[5]: import numpy as np 

print (np.char.title('i like runoob'))

Out[5]: I Like Runoob

**实例6**

numpy.char.lower() 函数对数组的每个元素转换为小写。它对每个元素调用 str.lower。

In[6]: import numpy as np 

#操作数组

print (np.char.lower(['RUNOOB','GOOGLE']))
 
# 操作字符串

print (np.char.lower('RUNOOB'))

Out[6]: ['runoob' 'google']

runoob

**实例7**

numpy.char.upper() 函数对数组的每个元素转换为大写。它对每个元素调用 str.upper。

In[7]: import numpy as np 

#操作数组

print (np.char.upper(['runoob','google']))
 
# 操作字符串

print (np.char.upper('runoob'))

Out[7]: ['RUNOOB' 'GOOGLE']

RUNOOB

**实例8**

numpy.char.split() 通过指定分隔符对字符串进行分割，并返回数组。默认情况下，分隔符为空格。

In[8]: import numpy as np 

# 分隔符默认为空格

print (np.char.split ('i like runoob?'))

# 分隔符为 .

print (np.char.split ('www.runoob.com', sep = '.'))

Out[8]: ['i', 'like', 'runoob?']

['www', 'runoob', 'com']

**实例9**

numpy.char.splitlines() 函数以换行符作为分隔符来分割字符串，并返回数组。

In[9]: import numpy as np 

# 换行符 \n   \n，\r，\r\n 都可用作换行符。

print (np.char.splitlines('i\nlike runoob?')) 

print (np.char.splitlines('i\rlike runoob?'))

Out[9]: ['i', 'like runoob?']

['i', 'like runoob?']

**实例10**

numpy.char.strip() 函数用于移除开头或结尾处的特定字符。

In[10]: import numpy as np 

# 移除字符串头尾的 a 字符

print (np.char.strip('ashok arunooba','a'))
 
# 移除数组元素头尾的 a 字符

print (np.char.strip(['arunooba','admin','java'],'a'))

Out[10]: shok arunoob

['runoob' 'dmin' 'jav']

**实例11**

numpy.char.join() 函数通过指定分隔符来连接数组中的元素或字符串

In[11]: import numpy as np 

# 操作字符串

print (np.char.join(':','runoob'))
 
# 指定多个分隔符操作数组元素

print (np.char.join([':','-'],['runoob','google']))

Out[11]: r:u:n:o:o:b

['r:u:n:o:o:b' 'g-o-o-g-l-e']

**实例12**

numpy.char.replace() 函数使用新字符串替换字符串中的所有子字符串。

In[12]: import numpy as np 

print (np.char.replace ('i like runoob', 'oo', 'cc'))

Out[12]: i like runccb

**2、数学函数**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Numpy 包含大量的各种数学运算的函数，包括三角函数，算术运算的函数，复数处理函数等。

**实例1**

Numpy 提供了标准的三角函数：sin()、cos()、tan()。

In[1]: import numpy as np 

a = np.array([0,30,45,60,90])

print ('不同角度的正弦值：')

# 通过乘 pi/180 转化为弧度  

print (np.sin(a*np.pi/180))

print ('\n')

print ('数组中角度的余弦值：')

print (np.cos(a*np.pi/180))

print ('\n')

print ('数组中角度的正切值：')

print (np.tan(a*np.pi/180))

Out[1]: 不同角度的正弦值：

[0.         0.5        0.70710678 0.8660254  1.        ]


数组中角度的余弦值：

[1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]


数组中角度的正切值：

[0.00000000e+00 5.77350269e-01 1.00000000e+00 1.73205081e+00 1.63312394e+16]

**实例2**

arcsin，arccos，和 arctan 函数返回给定角度的 sin，cos 和 tan 的反三角函数。

这些函数的结果可以通过 numpy.degrees() 函数将弧度转换为角度。

In[2]: import numpy as np 

a = np.array([0,30,45,60,90])  

print ('含有正弦值的数组：')

sin = np.sin(a*np.pi/180)  

print (sin)

print ('\n')

print ('计算角度的反正弦，返回值以弧度为单位：')

inv = np.arcsin(sin)  

print (inv)

print ('\n')

print ('通过转化为角度制来检查结果：')

print (np.degrees(inv))

print ('\n')

print ('arccos 和 arctan 函数行为类似：')

cos = np.cos(a*np.pi/180)  

print (cos)

print ('\n')

print ('反余弦：')

inv = np.arccos(cos)  

print (inv)

print ('\n')

print ('角度制单位：')

print (np.degrees(inv))

print ('\n')

print ('tan 函数：')

tan = np.tan(a*np.pi/180)  

print (tan)

print ('\n')

print ('反正切：')

inv = np.arctan(tan)  

print (inv)

print ('\n')

print ('角度制单位：')

print (np.degrees(inv))

Out[2]: 含有正弦值的数组：

[0.         0.5        0.70710678 0.8660254  1.        ]


计算角度的反正弦，返回值以弧度为单位：

[0.         0.52359878 0.78539816 1.04719755 1.57079633]



通过转化为角度制来检查结果：

[ 0. 30. 45. 60. 90.]

arccos 和 arctan 函数行为类似：

[1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01 6.12323400e-17]


反余弦：

[0.         0.52359878 0.78539816 1.04719755 1.57079633]


角度制单位：

[ 0. 30. 45. 60. 90.]


tan 函数：

[0.00000000e+00 5.77350269e-01 1.00000000e+00 1.73205081e+00 1.63312394e+16]


反正切：

[0.         0.52359878 0.78539816 1.04719755 1.57079633]


角度制单位：

[ 0. 30. 45. 60. 90.]

**实例3**

numpy.around() 函数返回指定数字的四舍五入值。

numpy.around(a,decimals)

In[3]: import numpy as np 

a = np.array([1.0,5.55,  123,  0.567,  25.532]) 

print  ('原数组：')

print (a)

print ('\n')

print ('舍入后：')

print (np.around(a))

print (np.around(a, decimals =  1))

print (np.around(a, decimals =  -1))

Out[3]: 原数组：

[  1.      5.55  123.      0.567  25.532]


舍入后：

[  1.   6. 123.   1.  26.]

[  1.    5.6 123.    0.6  25.5]

[  0.  10. 120.   0.  30.]

**实例4**

numpy.floor() 返回小于或者等于指定表达式的最大整数，即向下取整。

In[4]: import numpy as np 

a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print ('提供的数组：')

print (a)

print ('\n')

print ('修改后的数组：')

print (np.floor(a))

Out[4]: 提供的数组：

[-1.7  1.5 -0.2  0.6 10. ]


修改后的数组：

[-2.  1. -1.  0. 10.]

**实例5**

numpy.ceil() 返回大于或者等于指定表达式的最小整数，即向上取整。

In[5]: import numpy as np 

a = np.array([-1.7,  1.5,  -0.2,  0.6,  10]) 

print  ('提供的数组：')

print (a)

print ('\n')

print ('修改后的数组：')

print (np.ceil(a))

Out[5]: 提供的数组：

[-1.7  1.5 -0.2  0.6 10. ]


修改后的数组：

[-1.  2. -0.  1. 10.]

**3、算数函数**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Numpy 算术函数包含简单的加减乘除: add()，subtract()，multiply() 和 divide()。

需要注意的是数组必须具有相同的形状或符合数组广播规则。

**实例1**

In[1]: import numpy as np 

a = np.arange(9, dtype = np.float_).reshape(3,3) 

print ('第一个数组：')

print (a)

print ('\n')

print ('第二个数组：')

b = np.array([10,10,10])  

print (b)

print ('\n')

print ('两个数组相加：')

print (np.add(a,b))

print ('\n')

print ('两个数组相减：')

print (np.subtract(a,b))

print ('\n')

print ('两个数组相乘：')

print (np.multiply(a,b))

print ('\n')

print ('两个数组相除：')

print (np.divide(a,b))

Out[1]: 第一个数组：

[[0. 1. 2.]

 [3. 4. 5.]

 [6. 7. 8.]]


第二个数组：

[10 10 10]


两个数组相加：

[[10. 11. 12.]

 [13. 14. 15.]

 [16. 17. 18.]]


两个数组相减：

[[-10.  -9.  -8.]

 [ -7.  -6.  -5.]

 [ -4.  -3.  -2.]]


两个数组相乘：

[[ 0. 10. 20.]

 [30. 40. 50.]

 [60. 70. 80.]]


两个数组相除：

[[0.  0.1 0.2]

 [0.3 0.4 0.5]

 [0.6 0.7 0.8]]

**实例2**

numpy.reciprocal() 函数返回参数逐元素的倒数。如 1/4 倒数为 4/1。

In[2]: import numpy as np 

a = np.array([0.25,  1.33,  1,  100])  

print ('我们的数组是：')

print (a)

print ('\n')

print ('调用 reciprocal 函数：')

print (np.reciprocal(a))

Out[2]: 我们的数组是：

[  0.25   1.33   1.   100.  ]


调用 reciprocal 函数：

[4.        0.7518797 1.        0.01     ]

**实例3**

numpy.power() 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。

In[3]: import numpy as np 

a = np.array([10,100,1000]) 

print ('我们的数组是；')

print (a)

print ('\n') 

print ('调用 power 函数：')

print (np.power(a,2))

print ('\n')

print ('第二个数组：')

b = np.array([1,2,3])  

print (b)

print ('\n')

print ('再次调用 power 函数：')

print (np.power(a,b))

Out[3]: 我们的数组是；

[  10  100 1000]


调用 power 函数：

[    100   10000 1000000]


第二个数组：

[1 2 3]


再次调用 power 函数：

[        10      10000 1000000000]

**实例4**

numpy.mod() 计算输入数组中相应元素的相除后的余数。 函数 numpy.remainder() 也产生相同的结果。

In[4]: import numpy as np 

a = np.array([10,20,30]) 

b = np.array([3,5,7]) 

print ('第一个数组：')

print (a)

print ('\n')

print ('第二个数组：')

print (b)

print ('\n')

print ('调用 mod() 函数：')

print (np.mod(a,b))

print ('\n')

print ('调用 remainder() 函数：')

print (np.remainder(a,b))

Out[4]: 第一个数组：

[10 20 30]


第二个数组：

[3 5 7]


调用 mod() 函数：

[1 0 2]


调用 remainder() 函数：

[1 0 2]

**4、统计函数**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**实例1**

numpy.amin() 用于计算数组中的元素沿指定轴的最小值。

numpy.amax() 用于计算数组中的元素沿指定轴的最大值。

In[1]: import numpy as np 

a = np.array([[3,7,5],[8,4,3],[2,4,9]])  

print ('我们的数组是：')

print (a)

print ('\n')

print ('调用 amin() 函数：')

print (np.amin(a,1))

print ('\n')

print ('再次调用 amin() 函数：')

print (np.amin(a,0))

print ('\n')

print ('调用 amax() 函数：')

print (np.amax(a))

print ('\n')

print ('再次调用 amax() 函数：')

print (np.amax(a, axis =  0))

Out[1]: 我们的数组是：

[[3 7 5]

 [8 4 3]

 [2 4 9]]


调用 amin() 函数：

[3 3 2]


再次调用 amin() 函数：

[2 4 3]


调用 amax() 函数：

9


再次调用 amax() 函数：

[8 7 9]

**实例2**

numpy.ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值）。

In[2]: import numpy as np 

a = np.array([[3,7,5],[8,4,3],[2,4,9]])  

print ('我们的数组是：')

print (a)

print ('\n')

print ('调用 ptp() 函数：')

print (np.ptp(a))

print ('\n')

print ('沿轴 1 调用 ptp() 函数：')

print (np.ptp(a, axis =  1))

print ('\n')

print ('沿轴 0 调用 ptp() 函数：')

print (np.ptp(a, axis =  0))

Out[2]: 我们的数组是：

[[3 7 5]

 [8 4 3]

 [2 4 9]]


调用 ptp() 函数：

7


沿轴 1 调用 ptp() 函数：

[4 5 7]


沿轴 0 调用 ptp() 函数：

[6 3 6]

**实例3**

百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。 函数numpy.percentile()接受以下参数。

numpy.percentile(a, q, axis)

In[3]: import numpy as np 

a = np.array([[10, 7, 4], [3, 2, 1]])

print ('我们的数组是：')

print (a)
 
print ('调用 percentile() 函数：')

# 50% 的分位数，就是 a 里排序之后的中位数

print (np.percentile(a, 50)) 
 
# axis 为 0，在纵列上求

print (np.percentile(a, 50, axis=0)) 
 
# axis 为 1，在横行上求

print (np.percentile(a, 50, axis=1)) 
 
# 保持维度不变

print (np.percentile(a, 50, axis=1, keepdims=True))

Out[3]: 我们的数组是：

[[10  7  4]

 [ 3  2  1]]

调用 percentile() 函数：

3.5

[6.5 4.5 2.5]

[7. 2.]

[[7.]

 [2.]]

**实例4**

numpy.median() 函数用于计算数组 a 中元素的中位数（中值）

In[4]: import numpy as np 

a = np.array([[30,65,70],[80,95,10],[50,90,60]]) 

print ('我们的数组是：')

print (a)

print ('\n')

print ('调用 median() 函数：')

print (np.median(a))

print ('\n')

print ('沿轴 0 调用 median() 函数：')

print (np.median(a, axis =  0))

print ('\n')

print ('沿轴 1 调用 median() 函数：')

print (np.median(a, axis =  1))

Out[4]: 我们的数组是：

[[30 65 70]

 [80 95 10]

 [50 90 60]]


调用 median() 函数：

65.0


沿轴 0 调用 median() 函数：

[50. 90. 60.]


沿轴 1 调用 median() 函数：

[65. 80. 60.]

**实例5**

numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。

算术平均值是沿轴的元素的总和除以元素的数量。

In[5]: import numpy as np 

a = np.array([[1,2,3],[3,4,5],[4,5,6]])  

print ('我们的数组是：')

print (a)

print ('\n')

print ('调用 mean() 函数：')

print (np.mean(a))

print ('\n')

print ('沿轴 0 调用 mean() 函数：')

print (np.mean(a, axis =  0))

print ('\n')

print ('沿轴 1 调用 mean() 函数：')

print (np.mean(a, axis =  1))

Out[5]: 我们的数组是：

[[1 2 3]

 [3 4 5]

 [4 5 6]]


调用 mean() 函数：

3.6666666666666665


沿轴 0 调用 mean() 函数：

[2.66666667 3.66666667 4.66666667]


沿轴 1 调用 mean() 函数：

[2. 4. 5.]

**实例6**

numpy.average() 函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。

In[6]: import numpy as np 

a = np.array([1,2,3,4])  

print ('我们的数组是：')

print (a)

print ('\n')

print ('调用 average() 函数：')

print (np.average(a))

print ('\n')

# 不指定权重时相当于 mean 函数

wts = np.array([4,3,2,1])  

print ('再次调用 average() 函数：')

print (np.average(a,weights = wts))

print ('\n')

# 如果 returned 参数设为 true，则返回权重的和  

print ('权重的和：')

print (np.average([1,2,3,  4],weights =  [4,3,2,1], returned =  True))

Out[6]: 我们的数组是：

[1 2 3 4]


调用 average() 函数：

2.5


再次调用 average() 函数：

2.0


权重的和：

(2.0, 10.0)

**实例7**

标准差是一组数据平均值分散程度的一种度量。

std = sqrt(mean((x - x.mean())**2))

In[7]: import numpy as np 

print (np.std([1,2,3,4]))

Out[7]: 1.1180339887498949

**实例8**

统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数，即 mean((x - x.mean())** 2)。

In[8]: import numpy as np 

print (np.var([1,2,3,4]))

Out[8]: 1.25