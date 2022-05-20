#!/usr/bin/env python
# coding: utf-8

# # 02.파이썬라이브러리-Pandas

# ## pandas 설치하기

# In[ ]:


get_ipython().system('pip install pandas')


# ## 1. pandas 소개

# ### pandas 사용

# In[2]:


import pandas as pd


# ### Pandas에서 사용하는 데이터 구조
#  - Series 객체: 1차원 배열, 1차원 ndarray와 호환
#  - DataFrame 객체: 2차원 배열. 서로 다른 자료형을 사용 가능. 행(인스턴스), 열(컬럼, 피쳐(feature))

# ### Series 객체

# #### - Series 정의하기

# In[6]:


data = pd.Series([4, 7, -5, 3])
data


# In[8]:


print(data.values)
print(data.index)
print(data.dtype)


# #### - Series  인덱스 정의하기

# In[9]:


data = pd.Series([4, 7, -5, 3], index=['a','b','c','d'])
data


# #### - Series 인덱스 변경하기

# In[10]:


data.index = ['A','B','C','D']
data


# #### - python의 dictionary 자료형을 Series data로 만들 수 있다

# In[12]:


# dictionary의 key가 Series의 index가 된다
dicData = {'국어': 90, '영어': 85, '수학': 95, '과학': 75}
data = pd.Series(dicData)
data


# In[13]:


data.index.name = '과목'
data.name = '성적'
data


# #### - 시리즈 인덱싱

# In[41]:


data = pd.Series({'국어': 90, '영어': 85, '수학': 95, '과학': 75})

data[1:2]


# In[42]:


data['영어':'과학']


# In[43]:


data[(data > 90)]


# #### - 시리즈 연산 : value에만 적용됨

# In[40]:


data1 = pd.Series({'국어': 90, '영어': 85, '수학': 95, '과학': 75})
data2 = pd.Series({'국어': 80, '영어': 95, '수학': 85, '과학': 75})

data1 - data2


# #### - 시리즈에 요소 추가 & 삭제

# In[44]:


data = pd.Series({'국어': 90, '영어': 85, '수학': 95, '과학': 75})

# 요소 추가
data['미술'] = 100

# 요소 삭제
del data['영어']

data


# ### DataFrame 객체

# #### - 2차원 리스트, 배열을 사용하여 생성

# In[15]:


df = pd.DataFrame([[1,2,3],
                   [4,5,6],
                   [7,8,9]])




# In[17]:


# DataFrame 이름 부여하기
df.name = 'test'
df.index.name = 'index'
df.columns.name = 'num'
df


# In[18]:


# index명, column명 부여하기
df.index = ['a','b','c']
df.columns = ['c1','c2','c3']
df


# #### - Python Dictionary 자료형을 사용하여 생성

# In[4]:


score_table = {'성명':['BTS','아이유','원빈','블랙핑크','나'],
              '영어':[50,60,70,80,90],
              '수학':[60,50,70,70,90]}

df = pd.DataFrame(score_table)
df


# In[5]:


df = pd.DataFrame(score_table, columns=["영어", "수학", "성명"],
                          index=["one", "two", "three", "four", "five"])
df


# In[25]:


df.name = 'Grade Card'
df.index.name = 'name'
df.columns.name = 'subject'
df


# In[7]:


df = pd.DataFrame(score_table, columns=["성명", "영어", "수학"],
                          index=["one", "two", "three", "four", "five"])
df


# #### - DafaFrame 복사하기

# In[8]:


df2 = df             # view, 메모리 공유
df3 = df.copy()      # copy, 별도의 메모리 사용
df4 = df.iloc[:,0:2] # copy, 별도의 메모리 사용
df4


# #### - DataFrame의 Column(열)에 접근하기

# In[9]:


df['성명']


# In[10]:


print(df[['성명', '영어']])


# #### - DataFrame의 Row(행)에 접근하기

# In[11]:


# 특정 개수만큼 행 출력
df.head(3)   # df.head(n=3)


# In[26]:


# 특정 조건으로 행 검색
df.query('성명=="BTS"')


# In[28]:


df.query('영어 > 70')['성명']


# #### -열 데이터 변경

# In[48]:


import numpy as np

df['영어'] = np.mean(df['영어'])

df['영어']


# #### - 열 추가하기

# In[50]:


df['국어'] = np.random.randint(100)
df


# #### - 열 추가하기: Series 만들어서 추가하기

# In[51]:


# Series를 추가할 수도 있다
val = pd.Series([80, 70, 90], index=['two','four','five'])

df['과학'] = val
df


# #### - 행 추가하기: 열에 값 지정해서 새로운 열 추가하기

# In[52]:


df.loc['six',:] = [90,90,'유재석',100,100]
df


# #### - 열 삭제하기

# In[53]:


del df['과학']
df


# #### - 행(특정 행) 삭제하기

# In[55]:


df.drop(['six'])


# ### 실습예제: 외부 데이터 읽어오기

# #### - csv 파일 입력
# 맥은 utf-8 방식을 쓰고 윈도우는 cp949 방식 사용
# 한글 인코딩: Microsoft사에서 만든 cp949/ms949 인코딩, euc-kr인코딩, utf-8 인코딩 등등
#  - 해결책 (1) - engine='python'
#  - 해결책 (2) - encoding='utf-8' 
#  - 해결책 (3) - encoding='cp949'  # 한글이 깨지거나 읽을 때 오류가 발생한다면
#  - 해결책 (4) - Excel에서 인코딩 옵션 변경 (다른 이름으로 저장에서 - CSV UTF-8 (쉼표로 분리) 로 변경하여 저장)

# In[11]:





# In[ ]:




