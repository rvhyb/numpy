import xmlrpc.client
import matplotlib.pyplot as plt
import numpy as np
a=np.array([4,5,6])
type(a)#输出a的类型
a.shape#维数大小numpy.ndarray
a[0]
b=np.array([[4,5,6],[1,2,3]])
print(b)
b.shape#输出维数大小（2，3）
c=np.zeros((3,3),dtype=int)#全为0的矩阵大小3*3
d=np.ones((4,5),dtype=int)#全为1的矩阵大小为4*5
m=np.identity(4)#单位矩阵大小4*4
n=np.random.randn(3,2)#随机数矩阵
q=np.arange(1,13).reshape(3,4)#输出3*4的矩阵元素为1到12
p=q[0:2,1:3]#将q中1到2行和2到3列放入p中
w=q[1:3,:]#将q中2到3行放入w中
e=np.array([[1,2],[3,4],[5,6]])
e[[0,1,2],[0,1,0]]#[1,4,5]输出(0,0),(1,1),(2,0)三个元素
r=np.arange(1,13).reshape(4,3)
t=np.array([0,2,0,1])
r[np.arange(4),t]#输出r中（0，0）（1，2）（2，0）（3，1）四个元素
r[np.arange(4),t]+=10#每个元素加10
y=np.array([1,2])
y.dtype#输出int32
u=np.array([1.0,2.0])
u.dtype#输出float64
i=np.array([[1,2],[3,4]],dtype=np.float64)
o=np.array([[5,6],[7,8]],dtype=np.float64)
i+o
np.add(i,o)#输出[[6.,8.],[10.,12.]]
i-o
np.subtract(i,o)#输出[[-4.,-4.],[-4.,-4.]]
i*o
np.multiply(i,o)#输出[[5.,12.],[21.,32.]]
np.dot(i,o)#满足矩阵相乘
#f=np.array([[2,3,4],[5,6,7]],dtype=int)
#g=np.array([[1,2],[7,8],[1,3]],dtype=int)
#np.dot(f,g)
#np.multiply(f,g)
i/o
np.divide(i,o)#俩矩阵下标相同的相除
np.sqrt(i)#矩阵元素开根号
i.dot(o)
np.dot(i,o)#俩运行结果相同，都是满足矩阵相乘定理
np.sum(i)#矩阵中所有元素相加
np.sum(i,axis=0)#列和 [4. 6.]
np.sum(i,axis=1)#行和 [3. 7.]
np.mean(i)#结果2.5平均值的求解
np.mean(i,axis=0)#列和的平均值，结果为[2. 3.]
np.mean(i,axis=1)#行和的平均值，结果为[1.5 3.5]\
i.T#矩阵的转置
np.exp(i)#e的指数[[2.71,7.38],[20.08,54.598]]
s=np.array([[7,8],[4,2]])
np.argmax(s)#输出1
np.argmax(s,axis=0)#输出[0 0]
np.argmax(s,axis=1)#输出[1 0]
x=np.arange(0,100,0.1)#建立一个0到100每个相邻元素相差0.1的矩阵
h=x*x
plt.figure(figsize=(6,6))#建立画布，并指定画布大小
plt.plot(x,h)#在画布上画图
plt.show()#展示画图结果
b=np.arange(0,3*np.pi,0.1)
y1=np.sin(b)#建立sin矩阵
y2=np.cos(b)
plt.figure(figsize=(10,6))
plt.plot(b,y1,color='Red')#在画布上画图，颜色为红
plt.plot(b,y2,color='Blue')
plt.legend(['Sin','Cos'])#给俩条线做标记
plt.show()