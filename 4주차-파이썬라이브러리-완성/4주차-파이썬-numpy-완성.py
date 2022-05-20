#!/usr/bin/env python
# coding: utf-8

# # 01.파이썬라이브러리-Numpy

# ## numpy 설치하기

# In[ ]:


get_ipython().system('pip install numpy')


# ### numpy 데이터 적용 예
# - 이미지 파일의 데이터를 확인해 보자!
# - 단, 아래 코드는 opencv-python 라이브러리가 설치되어 있는 상태에서 실행해야 오류가 없다.

# In[1]:


import sys
import cv2

print('Hello OpenCV', cv2.__version__)

image = cv2.imread('image\cat.jpg')
print(image)
print(type(image))
print(image.dtype)

if image is None:
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow("image", image)
cv2.waitKey()


# ## 1. numpy 배열

# ###  numpy 사용

# In[2]:


import numpy as np

a = [0,1,2,3,4,5]
print(type(a), a)

na = np.array(a)      # np.array()로 ndarray오브젝트 생성
print(type(na), na)


# In[19]:


import numpy as np

arr = np.array([0,1,2,3,4,5])
print(type(arr))
print(arr)


# ### 1차원 배열 만들기

# In[3]:


na = np.array([0,1,2,3,4,5,6,7,8,9])
na
print(na)


# In[4]:


na = np.array(range(0,10,1))
print(na)


# In[5]:


na = np.arange(10)
print(na)


# ## 실습: 2차원 배열 만들기

# In[5]:


import numpy as np

arr = np.array([[10,20,30,40],[50,60,70,80]])

arr = np.array([[1,2,3,4],[5,6,7,8]])
print(arr*10)


# ### 2차원 배열

# In[6]:


na = np.array([[0,1,2],[3,4,5]])  # 2  x 3 array (괄호의 갯수)
print(na)


# ### 3차원 배열

# In[7]:


na = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])
print(na)


# ### 배열의 차원과 크기

# In[8]:


na1 = np.array([0,1,2,3,4,5,6,7,8,9])
na2 = np.array([[0,1,2],[3,4,5]])
na3 = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])

print(len(na1),len(na2),len(na3))    # 배열의 요소 개수
print(len(na1),len(na2[0]),len(na3[1][0])) 


# #### - ndim : 배열의 차원

# In[9]:


na3.ndim    #배열의 차원


# #### - shape : 배열의 크기

# In[10]:


na3.shape   #배열의 크기   (면, 행, 열) matrix


# #### - reshape : 배열의 크기 변경

# In[3]:


na = np.arange(12)
print(na)


# In[12]:


nb = na.reshape(3,4)
print(nb)


# In[13]:


nb = na.reshape(3,-1)
print(nb)


# In[14]:


nb = na.reshape(2,2,-1)  # -1 : 자동으로 값이 계산
print(nb)


# In[5]:


nb = na.reshape(2,-1,2)
print(nb)


# #### - flatten / ravel : 1차원 배열로 만듦

# In[6]:


nc = nb.flatten()
print(nc)


# In[7]:


nc = nb.ravel()
print(nc)


# In[9]:


nc = nb.reshape(-1)
print(nc)


# In[12]:


nc.dtype


# In[15]:


np.iinfo('int64')


# ### 배열의 인덱싱 & 슬라이싱

# In[19]:


na1 = np.array([0,1,2,3,4,5,6,7,8,9])
print(na1[2])
print(na1[1:3])
print(na1[::-1])
print('-'*20)
na2 = np.array([[0,1,2],[3,4,5]])
print(na2[0,0])
print(na2[-1,2])
print(na2[:,1])
print(na2[1,1:])
print('-'*20)
na3 = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])
print(na3[0,0])
print(na3[0,1,2])
print(na3[1,1:2,2])


# ## 2.배열의 생성과 변형

# ### 배열의 데이터 타입

# In[20]:


na = np.array([0,1,2,3,4,5,6,7,8,9])
na.dtype


# #### 1.배열은 한가지 데이터 타입만 가질 수 있다.

# In[21]:


na = np.array([0,1,2,3.5,4,5,6,7,8,9])
print(na)
print(na.dtype)   # float


# In[22]:


na = np.array([0,1,2,3.5,'4',5,6,7,8,9])
print(na)
print(na.dtype)   # 유니코드 문자열


# In[ ]:





# #### 2.데이터 타입 지정

# In[23]:


na = np.array([1, 2, 3], dtype='float')
print(na.dtype)
print(na)


# In[24]:


print(int is np.int)
print(float is np.float)
print(float is np.float64)
print(float == 'float64')


# In[25]:


# 모두 동일한 데이터 타입
na = np.array([1, 2, 3], dtype=np.int64) 
print(na.dtype) # int64 

na = np.array([1, 2, 3], dtype='int64') 
print(na.dtype) # int64 

na = np.array([1, 2, 3], dtype='i8') 
print(na.dtype) # int64


# In[26]:


na = np.array([1, 2, 3], dtype=np.float64) 
print(na.dtype) # float64 

na = np.array([1, 2, 3], dtype='float64') 
print(na.dtype) # float64 

na = np.array([1, 2, 3], dtype='f8') 
print(na.dtype) # float64


# In[27]:


na = np.array([True, False])
#na = np.array([True, False], dtype='int')
print(na.dtype, na)


# #### 3.숫자형이 취할 수 있는 범위 (최소값, 최대값)의 확인

#  - 정수 int, unit이 취할 수 있는 범위 
#  - np.iinfo()

# In[21]:


print(np.iinfo('int64'))


#  - 부동소수점 float이 취할 수 있는 범위
#  - np.finfo()

# In[29]:


print(np.finfo('float'))


# #### 4.데이터 타입 변경

# In[30]:


na = np.array([0,1,2,3,4])
print(na)
print(na.dtype)

na = na.astype(np.float)
print(na)
print(na.dtype)


# ### 다양한 배열생성 방법

# #### - zeros

# In[31]:


na = np.zeros(5)
print(na)


# In[32]:


na = np.zeros((2, 3), dtype=int)  # 튜플 타입으로 지정
print(na)

na = np.zeros((2, 3, 4))
print(na)


# In[33]:


na = np.zeros(5, dtype="U4")
print(na)


# In[22]:


import numpy as np
import cv2

image = np.zeros((300, 400), np.uint8)
image.fill(200)      # 또는 image[:] = 200

cv2.imshow("Window title", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[34]:


na[0] = 'a'
na[1] = 'ab'
na[2] = 'abc'
na[3] = 'abcd'
print(na)


# #### - ones

# In[35]:


na = np.ones(5)
print(na)

na = np.ones((2,3), dtype='i')
print(na)


# #### - zeros_like, ones_like: 다른 배열과 같은 크기의 배열 생성

# In[36]:


na = np.ones_like(na)  #
print(na)


# #### - empty : 배열만 생성 (값을 초기화 하지 않음)

# In[37]:


na = np.empty((3,1))  # 배열의 크기가 커지면 배열을 초기화하는데도 시간이 걸린다
print(na)


# #### - arange : range와 유사한 명령

# In[38]:


na = np.arange(10)
print(na)


# In[180]:


na = np.arange(2, 10, 2)  # start, end(포함안됨), step
print(na)


# #### - linspace: 선형구간 지정, logspace: 로그구간 지정

# In[182]:


na = np.linspace(0, 100, 5, dtype=int)
print(na)


# In[183]:


na = np.logspace(0.1, 1, 10)
print(na)


# In[1]:


import numpy as np

a = np.zeros((2,5), np.int)
b = np.ones((3,1), np.uint8)
c = np.empty((1,5), np.float)
d = np.full(5, 15, np.float32)

print(type(a), type(a[0]), type(a[0][0]))
print(type(b), type(b[0]), type(b[0][0]))
print(type(c), type(c[0]), type(c[0][0]))
print(type(d), type(d[0]) )
print('c 형태:', c.shape, '   d 형태:', d.shape)
print(a), print(b)
print(c), print(d)


# ### 배열의 연결

# #### - hstack : 행의 수가 같은 두 개 이상의 배열을 옆으로 연결

# In[26]:


na1 = np.ones((2,3), dtype=int)
na2 = np.zeros((2,2), dtype=int)
print(na1)
print(na2)
np.hstack([na1, na2])


# #### - vstack : 열의 수가 같은 두 개 이상의 배열을 위아래로 연결

# In[27]:


na1 = np.ones((2,3), dtype=int)
na2 = np.zeros((3,3), dtype=int)
print(na1)
print(na2)
np.vstack([na1, na2])


# #### - dstack : 행이나 열이 아닌 깊이(depth) 방향으로 배열을 합침, 가장 안쪽 원소의 차원 증가 

# In[28]:


na1 = np.ones((2,3), dtype=int)
na2 = np.zeros((2,3), dtype=int)
na3 = np.zeros((2,3), dtype=int)
na = np.dstack([na1, na2, na3])
print(na1)
print(na2)
print(na3)
print('-'*20)
print(na)
print(na.shape)


# #### - stack : axis(default = 0) 인수를 사용하여 사용자가 지정한 차원(축으로) 배열을 연결

# In[214]:


na = np.stack([na1, na2, na3])
print(na)
print(na.shape)


# In[215]:


na = np.stack([na1, na2, na3], axis=1)
print(na)
print(na.shape)


# #### - r_ : hstack 명령과 비슷, 배열을 좌우로 연결, 대괄호 [ ] 사용하는 메서드

# In[216]:


np.r_[np.array([1, 2, 3]), np.array([4, 5, 6])]


# #### - c_ : 배열의 차원을 증가시킨 후 좌우로 연결하는 메서드

# In[217]:


np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]


# #### - tile : 동일한 배열을 반복하여 연결

# In[221]:


a = np.array([[0, 1, 2], [3, 4, 5]])
print(a)
print(np.tile(a, 2))


# In[219]:


np.tile(a, (3, 2))


# In[ ]:


array([[   0.,    0.,    0.,    1.,    1.],
       [   0.,    0.,    0.,    1.,    1.],
       [   0.,    0.,    0.,    1.,    1.],
       [  10.,   20.,   30.,   40.,   50.],
       [  60.,   70.,   80.,   90.,  100.],
       [ 110.,  120.,  130.,  140.,  150.],
       [   0.,    0.,    0.,    1.,    1.],
       [   0.,    0.,    0.,    1.,    1.],
       [   0.,    0.,    0.,    1.,    1.],
       [  10.,   20.,   30.,   40.,   50.],
       [  60.,   70.,   80.,   90.,  100.],
       [ 110.,  120.,  130.,  140.,  150.]])


# In[239]:


na1 = np.zeros((3,3))
na2 = np.ones((3,2))
na3 = np.arange(10,160,10)
a = np.vstack([np.hstack([na1,na2]), na3.reshape(3,5)])
b = a
np.vstack([a,b])


# ## 3. 배열의 연산

# ### Q: numpy 배열을 이용하여 아래 z = x + y 를 코딩하기

# In[17]:


x = np.arange(1,10001)
y = np.arange(10001, 20001)
z = x + y
z


# In[246]:


x = np.arange(1,10001)
y = np.arange(10001, 20001)

z = np.zeros_like(x)
for i in range(len(x)):
    z[i] = x[i] + y[i]
print(z)


# ### 벡터화 연산

# In[257]:


x = np.array([0,1,2,3,4])
print(x+1)


# In[ ]:


print(2*x)


# In[249]:


a = np.array([1,2,3])
b = np.array([10,20,30])
print(2*a + b)


# In[250]:


print(a == 2)


# In[251]:


print(b > 10)


# In[256]:


print((a == 2) & (b > 10))


# ### Q: 벡터화 연산 문제

# In[33]:


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


# - 배열에서 3의 배수를 찾아라

# In[38]:


print(x[x%3==0])


#  - 이 배열에서 4로 나누면 1이 남는 수를 찾아라.

# In[35]:


print(x[x%4==1])


#  - 이 배열에서 3으로 나누면 나누어지고 4로 나누면 1이 남는 수를 찾아라.

# In[36]:


print(x[(x%3==0) & (x%4==1)])


# ### 브로드캐스팅
#  - 서로 다른 크기를 가진 두 배열의 사칙 연산을 지원함
#  - 크기가 작은 배열을 자동으로 반복 확장하여 크기가 큰 배열에 맞추는 방법

# In[270]:


x = np.vstack([range(7)[i:i + 3] for i in range(5)])
x


# In[272]:


y = np.arange(5)[:,np.newaxis]
y


# In[273]:


x+y


# In[274]:


y = np.arange(3)
x+y


# ## 4.기술 통계

# numpy는 아래와 같은 데이터 집합에 대해 간단한 통계를 계산하는 함수(통계량)를 제공한다.
#  - 데이터의 개수(count)
#  - 평균(mean, average): 평균을 통계용어->표본 평균(sample average, sample mean)
#  - 분산(variance): 표본 분산, 데이터와 표본 평균간의 거리의 제곱의 평균, 표본 분산이 작으면 데이터가 모여있는 것이고 크면 흩어져 있는 것
#  - 표준 편차(standard deviation) : 표본 분산의 양의 제곱근 값
#  - 최댓값(maximum) : 데이터 중에서 가장 큰 값
#  - 최솟값(minimum) : 데이터 중에서 가장 작은 값
#  - 중앙값(median) : 데이터를 크기대로 정렬하였을 때 가장 가운데에 있는 수
#  - 사분위수(quartile) : 데이터를 가장 작은 수부터 가장 큰 수까지 크기가 커지는 순서대로 정렬하였을 때 1/4, 2/4, 3/4 위치에 있는 수
#  
#  x={18,5,10,23,19,−8,10,0,0,5,2,15,8,2,5,4,15,−1,4,−7,−24,7,9,−6,23,−13}

# In[318]:


x = np.array([18,5,10,23,19,-8,10,0,0,5,2,15,8,2,5,4,15,-1,4,-7,-24,7,9,-6,23,-13])
x


# #### - 데이터의 개수(count) : len

# In[277]:


len(x)


# #### - 평균(mean, average)

# In[280]:


np.mean(x)


# #### - 분산(variance)

# In[281]:


np.var(x)


# #### - 표준 편차(standard deviation)

# In[282]:


np.std(x)


# #### - 최댓값(maximum)

# In[283]:


np.max(x)


# In[285]:


np.argmax(x)


# #### - 최솟값(minimum)

# In[286]:


np.min(x)


# In[287]:


np.argmin(x)


# #### - 중앙값(median)

# In[288]:


np.median(x)


#  #### - 사분위수(quartile)

# In[290]:


np.percentile(x, 0) # 최소값


# In[289]:


np.percentile(x, 25)  # 1사분위 수


# In[291]:


np.percentile(x, 50)  # 2사분위 수


# In[292]:


np.percentile(x, 75)  # 3사분위 수


# In[293]:


np.percentile(x, 100)  # 최대값


# #### 히스토그램 그리기

# In[298]:


bins = np.arange(-25,25,5)  # 도수분포구간
hist, bins = np.histogram(x, bins)
print (hist)
print (bins)


# In[319]:


import matplotlib.pyplot as plt
plt.hist(x)
plt.show()


# In[320]:


n, bins, patches = plt.hist(x, bins=10)
plt.show()


# In[321]:


patches


# ## 5. 기타 (난수 )

# ### 난수 발생

# #### - 시드 설정
# 인수로는 0과 같거나 큰 정수를 넣어준다

# In[299]:


np.random.seed(0)


# #### - 난수 발생 : rand
#  0과 1사이의 난수를 발생

# In[317]:


np.random.rand(5)


# #### - randn: 표준 정규 분포:

# In[311]:


np.random.randn(5) 


# #### - randint: 균일 분포의 정수 난수: random.randint(low, high=None, size=None)

# In[315]:


np.random.randint(10,size=5) 


# In[316]:


np.random.randint(10, 20, size=5) 


# #### - 데이터 순서 임의로 바꾸기 :  shuffle

# In[306]:


x = np.arange(10)
print(x)
np.random.shuffle(x)
print(x)


# #### - 데이터 샘플링 : choice
# 이미 있는 데이터 집합에서 일부를 무작위로 선택하는 것을 표본선택 혹은 샘플링(sampling)이라고 한다
# 
# numpy.random.choice(a, size=None, replace=True, p=None)
#  - a : 배열이면 원래의 데이터, 정수이면 arange(a) 명령으로 데이터 생성
#  - size : 정수. 샘플 숫자
#  - replace : 불리언. True이면 한번 선택한 데이터를 다시 선택 가능
#  - p : 배열. 각 데이터가 선택될 수 있는 확률

# In[307]:


np.random.choice(5, 5, replace=False)  # shuffle 명령과 같다.


# In[308]:


np.random.choice(5, 3, replace=False)  # 3개만 선택


# In[309]:


np.random.choice(5, 10)  # 반복해서 10개 선택


# In[310]:


np.random.choice(5, 10, p=[0.1, 0, 0.3, 0.6, 0])  # 선택 확률을 다르게 해서 10개 선택


# -------

# ## 실습: 주사위를 100번 던져서 나오는 숫자의 평균을 구하라.

# In[20]:


x = np.random.randint(1,7, size=100) 
x


# #### -정렬하기

# In[23]:


np.sort(x)


# #### - 요소의 개수 세기: bincount

# In[25]:


np.bincount(x)[1:]   # 각 요소의 개수 세기


# #### - 평균, 중위수

# In[26]:


print(np.mean(x))
# x = np.median(x)


# #### - 히스토그램

# In[27]:


import matplotlib.pyplot as plt
plt.hist(x)
plt.show()


# ### Q: 가격이 10,000원인 주식이 있다. 이 주식의 일간 수익률(%)은 기댓값이 0%이고 표준편차가 1%인 표준 정규 분포를 따른다고 하자. 
# 250일 동안의 주가를 무작위로 생성하라
# 

# In[331]:


np.random.seed(0)
x = np.random.randn(250) 
x = x + 10000
x.astype(dtype=int)


# In[ ]:




