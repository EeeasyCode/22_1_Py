#!/usr/bin/env python
# coding: utf-8

# # 1장-집합과 함수

# ## 1.1 집합

# ### 1. 집합의 의미

# ### 파이썬에서의 집합(set) 자료형
# - ① 집합 자료는 키만 모아 놓은 딕셔너리의 특수한 형태
# - ② 딕셔너리의 키는 중복되면 안 되므로 세트에 들어 있는 값은 항상 유일
# - ③ 중복된 키는 자동으로 하나만 남음
# 

# In[84]:


# 파이썬에서 여러 데이터를 담을 수 있는 자료형

l = list()
t = tuple()
d = dict()  
s = set()


# ### 집합의 정의와 사용

# In[85]:


mySet = set()      # 공집합
print(type(mySet))

mySet = {1,2,3,4,5}
print(type(mySet))


# In[86]:


mySet = {1,2,2,3,3,4,4,4,}
print(mySet)


# In[87]:


mySet = {1,2,2,3,4,3,4,5}
print(mySet)
print(len(mySet))  # 길이


# In[88]:


mySet = {'사과','포도','오렌지','포도','포도'}
print(f'길이={len(mySet)}, 원소={mySet}')


# In[89]:


myList = ['사과','포도','오렌지','포도','포도']
print(f'길이={len(myList)}, 원소={myList}')
print(set(myList))


# In[90]:


mySet = set('Hello')
print(mySet)


# ### 부분집합( ⊂ ) 확인 

# In[91]:


A = {2, 4, 6, 8, 10}
B = {1, 2, 3, 4, 5}
C = {2, 4, 8}
D = set()    # 공집합

print(B.issubset(A))   # B ⊂ A 부분집합 확인 함수 issubset()
print(C.issubset(A))   # C ⊂ A 
print(D.issubset(A))   # D ⊂ A 공집합도 부분집합
print(A.issubset(A))   # A ⊂ A 자기 자신도 부분집합


# ------------------------------------------

# ### 2. 여러 가지 집합

# ### 1. 교집합 : &, intersection()

# #### 리스트만 이용해서 교집합 만들기

# In[92]:


A = [2, 4, 6, 8, 10]
B = [1, 2, 3, 4, 5]
S  = []
 
for element in A:
    if element in B:
        S.append(element)

print(set(S))


# #### 집합 함수 이용해서 교집합 만들기

# In[93]:


A = {2, 4, 6, 8, 10}
B = {1, 2, 3, 4, 5}

print(A & B)             # A, B의 순서는 상관 없다.
print(A.intersection(B))


# ### 2. 합집합 : |, union()

# #### 리스트만 이용해서 합집합 만들기

# In[94]:


A = [2, 4, 6, 8, 10]
B = [1, 2, 3, 4, 5]
S = []

print(set(A + B))


# #### 집합 함수 이용해서 합집합 만들기

# In[95]:


A = {2, 4, 6, 8, 10}
B = {1, 2, 3, 4, 5}

print(A | B)       # A.union(B) 으로 대체 가능 (A, B의 순서는 상관 없다.)
print(A.union(B))
print(B.union(A))


# ### 3. 차집합 : -, difference()

# #### 리스트만 이용해서 차집합 만들기

# In[96]:


A = [2, 4, 6, 8, 10]
B = [1, 2, 3, 4, 5]
S  = []
 
for element in A:
    if element not in B:
        S.append(element)

print(set(S))


# #### 집합 함수 이용해서 차집합 만들기

# In[97]:


A = {2, 4, 6, 8, 10}
B = {1, 2, 3, 4, 5}

print(A - B) # A.difference(B) 으로 대체 가능
print(A.difference(B)) 

print(B - A) # B.difference(A) 으로 대체 가능
print(B.difference(A)) 


# ### 4. 여집합

# #### 리스트를 이용해서 A의 여집합 구하기

# In[98]:


U = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
A = [2, 4, 6, 8, 10]
S = []

for element in U:
    if element not in A:
        S.append(element)       

print(set(S))


# #### 집합 함수 이용해서 A의 여집합 만들기

# In[99]:


U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
A = {2, 4, 6, 8, 10}

print(U.difference(A))
print(U ^ A)


# ### [실습] 예제1-2 집합 구하기
# 전체집합이 U = {x|x는 자연수, x<=10}일 때, 두 집합 A={1,3,5,7}과 B={3,4,5,6}에 대해 다음 집합을 구하라.
# - a. A ∩ B 
# - b. A ∪ B
# - c. A - B
# - d. A ∩ Bc 

# In[100]:


U = set(list(range(1,11)))
A = {1,3,5,7}
B = {3,4,5,6}

# a. A ∩ B 
print(A & B)
print(A.intersection(B))

# b. A ∪ B
print(A | B)
print(A.union(B))

# c. A - B
print(A - B)
print(A.difference(B))

# d. A ∩ Bc 
print(A & (U - B))
print(A & (U ^ B))
print(A.intersection(U.difference(B)))


# ### [실습] 예제1-3 집합의 성질
# 전체집합이 U = {x|x는 자연수, x<=10}에 대해 부분집합 A, B, C를 다음과 같이 정의한다. 
# A={x|x는 홀수}, B={x|x는 소수}, C={x|x는 3의 배수}
# 다음 집합을 구하라.
# - a. (A ∩ B) ∪ C 
# - b. (A ∪ C) ∩ (B ∪ C)
# - c. (A ∪ B) ∩ C
# - d. (A ∩ C) ∪ (B ∩ C)

# In[101]:


import sympy

U = set(list(range(1,11)))
A = set([x for x in U if x%2!=0])  
B = set([x for x in U if sympy.isprime(x)]) 
C = set([x for x in U if x%3==0])

print(A)
print(B)
print(C)

print(f'(A ∩ B) ∪ C       = {(A & B) | C}')
print(f'(A ∪ C) ∩ (B ∪ C)= {(A | C) & (B | C)}')
print(f'(A ∪ B) ∩ C       = {(A | B) &  C}')
print(f'(A ∩ C) ∪ (B ∩ C) = {(A & C) | (B & C)}')


# #### [소수(prime number) 판별 알고리즘]

# In[102]:


def is_prime_number(x):
    # 1 not being a prime number, is ignored
    if x > 1:
        for i in range(2, int(x/2)+1):
            if (x % i) == 0:
                print(f"{x} is not a prime number")
                break
        print(f"{x} is a prime number")
    else:
         print(f"{x} is not a prime number")
        
x = 7
is_prime_number(x)


# ### 집합에 요소 추가하기

# In[103]:


A = {1,3,5,7}

# 요소 한개 추가하기
print('요소 한개 추가: add')
A.add(9)

print(A)


# In[104]:


A = {1,3,5,7}

# 요소 여러 개 추가하기
print('요소 여러개 추가:update')

A.update([9,11,13])
print(A)


# ### 집합에서 요소 삭제하기

# In[105]:


A = {1,3,5,7}

print('요소 한개 제거: remove')
A.remove(5)  # 5 제거하기

print(A)


# ## 1.2 함수

# ### 일차함수(linear function)
# 일차 함수 그래프로 표현하기

# In[109]:


import matplotlib.pyplot as plt


def draw_linear_graph(a=1, b=1):    
    X = list(range(-5,5,1))     # X축 값
    Y = [(a*x + b) for x in X] # Y축 값
    
    plt.plot(X, Y)        # 선 그래프 그리기
    plt.grid()            # grid표시
    plt.xlabel('X-axis')  # X축 이름
    plt.ylabel('Y-axis')  # Y축 이름
    title = f'Linear equation graph: y={str(a)}x+{str(b)}'
    plt.title(title)
    #pl.savefig('Linear_graph_1.png')


# y = ax + b
a = int(input('Enter the coefficient of x: '))
b = int(input('Enter the constant: '))

draw_linear_graph(a, b)


# ### 이차함수(quadratic function)

# In[107]:


import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x ** 2

x = [-1,0,1]
x = np.linspace(-1, 1, 10)

plt.plot(x, f(x), 'ro-', label="y=f(x)=x^2")   
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.title("function graph")
plt.legend()

plt.show()


# In[108]:


import matplotlib.pyplot as plt
import numpy as np

def draw_graph(a=1, b=1, c=1, xmin=1, xmax=5):
    X = np.arange(xmin,xmax,0.001)
    Y = [(a*x**2 + b*x + c) for x in X]
    
    plt.plot(X, Y)             # 선 그래프 그리기
    plt.grid()                 # grid표시
    plt.xlabel('X-axis')       # X축 이름
    plt.ylabel('Y-axis')       # Y축 이름
    title = f'Linear graph: y={str(a)}x^2+{str(b)}x+{str(c)}'
    plt.title(title)
    plt.show()

# y=ax^2 + bx + c
a = float(input('Enter the coefficient of x^2: '))
b = float(input('Enter the coefficient of x: '))
c = float(input('Enter the constant: '))

xmin = float(input('Enter the minimum of x-range: '))
xmax = float(input('Enter the maximum of x-range: '))
    
draw_graph(a, b, c, xmin, xmax)


# ## 1.3 경우의 수

# ### 1.합의 법칙

# ### [실습] 예제1-10 경우의 수 구하기
# 주사의를 두 번 던져서 나온 눈의 수의 합이 소수인 경우의 수 구하라.

# In[75]:


def is_prime_number(x):
    # 1 not being a prime number, is ignored
    if x > 1:
        for i in range(2, int(x/2)+1):
             if (x % i) == 0:
                return False
        return True
    else:
         return False
        

d1 = list(range(1,7))
d2 = list(range(1,7))
d = list()

for x in d1:
    for y in d2:
        if is_prime_number(x+y):
            d.append((x,y))
            
print(f'두 눈의 수의 합이 소수인 경우-> {d}')
print(f'경우의 수->{len(d)}')


# In[76]:


import sympy

d1 = list(range(1,7))
d2 = list(range(1,7))

d = [(x,y) for x in d1 for y in d2 if sympy.isprime(x+y)]

print(f'두 눈의 수의 합이 소수인 경우-> {d}')
print(f'경우의 수->{len(d)}')


# ### [실습] 4개 문자 A, B, C, D 나열하는 경우의 수
# - 반복(x) + 순서(O) : permutations()
# - 반복(x) + 순서(x) : combinations()
# - 반복(O) + 순서(O) : product( )
# - 반복(O) + 순서(O) : combination_with_replacement()

# #### 1. 4개의 문자를 서로 다르게 나열하는 방법의 수
# - 반복(x) + 순서(O) : permutations()

# In[79]:


import itertools

S = {'A','B','C','D'}
r = 4

R = list(itertools.permutations(S, r))

print(f'순열 경우의 수: {len(R)}')


# In[80]:


R = list(itertools.permutations('ABCD', 4))

print(f'순열 경우의 수: {len(R)}')


# #### 2. 4개 중 서로 다른 문자 2개 순서없이 나열하는 방법의 수
# - 반복(x) + 순서(x) : combinations()

# In[82]:


import itertools

S = {'A','B','C','D'}
r = 2

R = list(itertools.combinations(S, r))

print(f'조합 경우의 수: {len(R)}')


# In[83]:


R = list(itertools.combinations('ABCD', r))

print(f'조합 경우의 수: {len(R)}')


# In[ ]:




