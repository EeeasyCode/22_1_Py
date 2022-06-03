#!/usr/bin/env python
# coding: utf-8

# # Chapter08 표본분포와 통계적 추정

# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # '맑은 고딕'으로 설정 
plt.rcParams['axes.unicode_minus'] = False     # 그래프에서 (-)숫자표시


# ## 8-1.모집단과 표본

# ### 모집단 분포와 표본분포

# ### [실습] 예제 8-1: 모수 구하기 (모평균, 모분산, 모표준편차 )
# 경부고속도록 서울 요금소~ 부산 구서 나들목에 있는 34개의 나들목 사이의 거리 측정 결과(소수점 셋째자리에서 반올림)

# In[ ]:



data =[9.59, 4.62, 0.65, 7.75, 16.98, 11.78, 7.24, 10.15, 25.49, 11.44, 10.37,
    9.33, 15.04, 12.16, 16.63, 12.06, 9.70, 12.46, 8.05, 19.91, 5.58, 12.48,
    4.35, 16.41, 22.53, 17.56, 18.4, 10.86, 27.43, 7.39, 14.57, 11.92, 2.00]  




print(f'a. 모평균   : {round( , 2)}')
print(f'b. 모분산   : {round( , 2)}')   
print(f'c. 모표준편차: {round( , 2)}')   


# ### [실습] 예제 8-2 : 표본 통계량 구하기
# sample = 7.75, 9.70, 10.86

# In[ ]:


sample = [7.75, 9.70, 10.86]



print(f'a. 표본평균   : {round( , 2)}')
print(f'b. 표본분산   : {round( , 2)}')   
print(f'c. 표본표준편차: {round( , 2)}') 


# ### 표본평균의 분포 - 균등분포

# #### # 경우의 수
# n=2인 표본평균의 경우의 수

# In[ ]:


import itertools

n = 2
result = list(itertools.product(([1,2,3,4]), repeat=n)) # 복원추출
print("**경우의 수 : %s개" % len(result))
print(result)


# #### # x_set: 확률변수 상태공간

# In[ ]:


x_set = [ sum([j for j in i])/n for i in result] 
x_set = list(set(x_set))
x_set.sort()
x_set


# #### 확률질량함수: f(x)

# In[ ]:


def f(x):
    cnt = 0
    for i in result:
        if sum([j for j in i])/n == x:
            cnt += 1
    return cnt / (len(result))


# #### 확률변수: X

# In[ ]:


X = [x_set, f]


# #### 확률: P(prob)

# In[ ]:


prob = [f(x_k) for x_k in x_set]
prob


# #### 확률분포표 

# In[ ]:


import pandas as pd

df = pd.DataFrame([prob], columns=x_set, index=['P(X_=x_)'])
df.columns.names = ['X_']
df


# #### 평균의 기대값

# In[ ]:


import numpy as np

def E(X):
    x_set, f = X
    return np.sum([x_k * f(x_k) for x_k in x_set]) 

E(X)


# #### 표본평균의 분산

# In[ ]:


def V(X):
    x_set, f = X
    mean = E(X)
    return np.sum([(x_k - mean)**2 * f(x_k) for x_k in x_set])

V(X)


# #### 이항균등분포의 표본평균의 분포 그래프로 나타내기

# In[ ]:


def get_sample_dist(X, n):
    
    import itertools
    result = list(itertools.product((X), repeat=n)) # 복원추출
    print(f'**경우의 수 : {len(result)}개')
    print(f'**모든 경우 : {result}')


    # 확률변수 상태공간
    x_set = [ sum([j for j in i])/n for i in result] 
    x_set = list(set(x_set))
    x_set.sort()
    print(f'**모든 확률변수 : {x_set}')

    # 확률질량함수
    def f(x):
        cnt = 0
        for i in result:
            if sum([j for j in i])/n == x:
                cnt += 1
        return cnt / (len(result))

    prob = [f(x_k) for x_k in x_set]
    print(f'**모든 확률   : {prob}')


    # 확률분포표
    df = pd.DataFrame([prob], columns=x_set, index=['P(X_=x_)'])
    df.columns.names = ['X_']
    print(f'**확률분포표:\n/{df}')

    return x_set, prob, df


# #### 표본평균의 확률분포 그래프

# In[ ]:


#---------------------
# 그래프로 나타내기 
#---------------------
plt.figure(figsize=(10,6))

X = [1,2,3,4]
for idx, n in enumerate(X): 
    x_set, prob, df = get_sample_dist(X, n)   

    plt.subplot(2, 2, idx+1) 
    plt.plot(x_set, prob, 'o-')
    plt.title(f'n={n}')
    
plt.show() 


# ### [실습] 예제 8-4 : 표본평균의 분포 및 확률 구하기
# $N(178, 16)$인 모집단에서 크기 9인 표본 선정, 표본평균 $ \bar{X}$
# - a. $\bar{X}$ 분포
# - b. $P(\bar{X} \le 180)$ 
# - c. $P(176 \le \bar{X} \le 180)$
# 

# In[ ]:







# #### [실습] $N(166, 9)$인 모집단에서 크기 16인 표본 선정, 표본평균 $ \bar{X}$
# - a. $\bar{X}$ 분포
# - b. $P(\bar{X} \le 180)$ 
# - c. $P(176 \le \bar{X} \le 180)$
# 

# In[ ]:







# ### [실습] 예제 8-5 : 표본비율의 분포 및 확률 구하기
# $p=0.45$인 모집단에서 크기 100인 표본 선정, 표본비율 $ \hat{p}$
# - a. $ \hat{p}$ 분포
# - b. $P(\hat{p} \le 0.35)$ 
# - c. $P(0.41 \le \hat{p} \le 0.51)$

# In[ ]:









# ----------------------------------------------------------

# ## 8-2. 모평균의 추정

# ### 모평균의 점추정

# In[ ]:


import numpy as np

data = [17.4, 17.2, 18.1, 17.5, 17.7,
       17.6, 17.5, 17.1, 17.8, 17.6]
N = np.array(data)

print(f'a. 표본평균   :  {round( , 4)}')
print(f'a. 표본분산   :  {round( , 4)}')
print(f'a. 표본표준편차: {round( , 4)}')


# ### 모평균의 신뢰구간

# #### 모분산이 알려진 정규모집단의 모평균에 대한 신뢰구간

# ### [실습] 예제 8-7 : 모분산이 알려진 정규모집단의 신뢰구간

# In[ ]:


Z = {  }


# In[ ]:


n, x_, var = 25, 30, 9
print(f'n, x_, var : {n, x_, var}')

a =  
b =  
print(f'모평균의 신뢰구간 : {a} <= mu <= {b}')


# #### 모분산이 알려지지 않은 정규모집단의 신뢰구간(근사확률분포)

# ### [실습] 예제 8-8 : 모분산이 알려지지 않은 정규모집단의 신뢰구간

# In[ ]:



   
   
   


# ### [실습] 예제 8-9 : 두 모평균 차의 신뢰구간

# In[ ]:


n, x_, sigma1 =  
m, y_, sigma2 =  
print(f'n, x_, sigma1 : {n, x_, sigma1}')
print(f'm, y_, sigma2 : {m, y_, sigma2}')

a =  
b =  
print(f'두 모평균 차의 신뢰구간 : {round(a,2)} <= x_ - y_ <= {round(b,2)}')


# ----------------------------------------------------------

# ## 8-3. 모비율의 추정

# ### 모비율의 신뢰구간

# ### [실습] 예제 8-10 : 모비율의 신뢰구간

# In[ ]:


Z = {90:1.645, 95:1.96, 99:2.58}

n, p, q =  
print(f'n, p, q : {n, p, q}')

a =   
b =  
print(f'모비율의 신뢰구간 : {round(a, 2)} <= p^ <= {round(b, 2)}')


# ### [실습] 예제 8-11 : 두 모비율 차의 신뢰구간

# In[ ]:


n, m  =  
p1,p2 =  
q1,q2 =  
print(f'n, p1, q1 : {n, p1, q1}')
print(f'm, p2, q2 : {n, p2, q2}')

a =  
b =  
print(f'두 모비율 차의 신뢰구간 : {round(a,4)} <= p1-p2 <= {round(b,4)}')


# ----------------------------------------------------------

# 끝
