#!/usr/bin/env python
# coding: utf-8

# # Chapter 6 확률변수

# In[32]:


get_ipython().run_line_magic('precision', '3')


# ## 6-1. 이산확률변수

# ### [실습] 예제 6-1: 확률변수 X의 상태공간(Sx) 구하기
# 주사위 두 번 던지기. 확률변수 X = |i - j|, Sx = ?

# In[2]:


X = list()
for i in range(1,7):
    for j in range(1, 7):
        X.append( abs(i-j)  )
Sx = set(X)
print(f'확률변수 X의 상태공간(Sx) = {Sx}')


# In[20]:


import random

N = 100
X = list()
for k in range(N):
    i = random.randint(1, 6)
    j = random.randint(1, 6)
    X.append( abs(i-j) )
#     print(f'시행횟수({k+1:>3}) - D1:{i}, D2:{j}, [i-j]:{abs(i-j)} ')
# print(X)
Sx = set(X)
print(f'확률변수 X의 상태공간(Sx) = {Sx}')


# ### [확률변수 예] 붓꽃데이터 가져오기
# -꽃잎(sepal), 꽃받침(petal)의 length, width, 붓꽃의 종류(species)가 모두 확률변수로 사용될 수 있다.

# In[23]:


# 통계 시각화 라이브러리
get_ipython().system('pip install seaborn')


# In[26]:


import seaborn as sns

iris = sns.load_dataset('iris')
print( type(iris) )
iris.head(3)


# ### [실습] 예제6-3 확률질량함수로 확률구하기
# 주사위 두 번 던지기. 확률변수 X = |i - j|, Sx 원소에 대한 확률함수
# 

# ##### 확률질량함수(PMF: probability mass function) 
# 이산확률변수의 확률분포를 나타내는 함수 <br><br>

# - 1. 확률변수 X에 대한 상태공간 구하기

# In[28]:


# 1. 확률변수 X에 대한 상태공간 구하기
X = list()   # 확률변수
D1 = [1,2,3,4,5,6]
D2 = [1,2,3,4,5,6]
for i in D1:
    for j in D2:
        X.append( abs(i-j)  )
Sx = set(X)
print(f'확률변수 X의 상태공간(Sx) = {Sx}')


# - 2. 확률변수에 대한 확률질량함수(fx) 만들기

# In[ ]:


# 소수점 자리수 표현
get_ipython().run_line_magic('precision', '3')


# In[52]:


# 2. 확률변수에 대한 확률질량함수(fx) 만들기
def f(x):
    cnt = 0
    for i in D1:
        for j in D2:
            if abs(i-j) == x:
                cnt += 1
            
    return cnt / (len(D1)*len(D2))

probs = [ f(x) for x in Sx ]
probs


# - **확률 전체의 합은 1**

# In[53]:


sum(probs)


# - 확률변수 X의 상태 값과 확률 함께 표시

# In[54]:


dict(zip(Sx, probs))


# - 3.확률변수 X와 확률의 관계를 그래프로 나타내기

# In[56]:


import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

X = list(Sx)
Y = probs
print(f'확률변수(X) : {X}')
print(f'확률(Y) : {Y}')

plt.bar(X, Y, zorder=1)
plt.scatter(X, Y, c='r', zorder=2)
for x, y in zip(X, Y):
    plt.text(x, y, (x, round(y,2)), fontsize=10)

plt.grid()
plt.title('확률변수X와 확률P')
plt.xlabel('확률변수X')
plt.ylabel('확률P(X)')
plt.show()


# In[ ]:




