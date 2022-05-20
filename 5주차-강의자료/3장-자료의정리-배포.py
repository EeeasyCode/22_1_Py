#!/usr/bin/env python
# coding: utf-8

# # Chapter03 자료의 정리

# ## 1. 자료의 종류

# **- 질적자료: 숫자에 의해 표현되지 않는 자료**
#   ex: 혈액형, 만족도, 학년별
#   
# **- 양적자료: 숫자로 표현되고, 그 숫자에 의미가 부여되는 자료**
#   ex: 스팸문자 횟수, 몸무게, 키
#   
#   1. 이산자료: 셈을 할 수 있는 자료 (스팸문자 횟수) 
#   2. 연속자료: 어떤 구간 안에서 측정되는 자료 (몸무게 , 키)

# ### 자료를 표현하는 방법

# ### [실습] [예제 3-2]

# #### 방법1: values(context), columns, index  지정해서 만들기

# In[ ]:





# #### 방법2: dictionary 이용

# In[ ]:






# ### B.그래프 그리기

# #### 1.선 그래프

# In[ ]:







# #### 2.점그래프

# In[ ]:






# #### 3.선+marker 그래프

# In[ ]:






# #### 4.막대 그래프

# In[ ]:






# #### 5. 원(파이) 그래프

# In[ ]:







# --------

# ## 2.질적자료의 정리

# In[169]:


import matplotlib.pyplot as plt

# 한글출력 설정
plt.rcParams['font.family'] = 'Malgun Gothic'# '맑은 고딕'으로 설정 
# 그래프 크기 지정
plt.rcParams['figure.figsize'] = (6, 4)
# 선 굵기 지정
plt.rcParams['lines.linewidth'] = 2 
#matplotlib.rcParams['axes.unicode_minus'] = False


# ### 점도표 만들기

# #### 산점도

# In[174]:


import matplotlib.pyplot as plt 
import numpy as np

index = ['매우만족','만족','보통','불만족','매우불만족']
data  = [5,11,21,9,4]

plt.scatter(index, data)  # index:x, data:y  
plt.show()


# ### [실습] Q. 만족도 점도표 표현하기

# In[204]:












# ### [실습] Q. 도수표 만들기

# #### Q.학년별 동아리 회원 수에 대한 도수표 작성하기

# In[209]:









# #### Q.학년별 동아리 회원 수에 대한 막대 그래프

# In[217]:







# ### [실습] 막대 그래프 & 꺽은선 그래프

# #### Q.성별에 따른 고객 만족도

# In[397]:


df  = pd.DataFrame([[3,7,10,4,2],[2,4,11,5,2]], 
                   index=['남자','여자'], 
                   columns=['매우만족','만족','보통','불만족','매우불만족'])
df


# #### 막대그래프

# In[250]:


# 막대그래프
x = np.arange(len(df.columns))
width = 0.3

plt.bar(x - width/2, df.loc['남자'], width,label='남자')
plt.bar(x + width/2, df.loc['여자'], width,label='여자')
plt.title('성별에 따른 고객 만족도')
plt.xlabel('만족도')
plt.ylabel('명')
plt.xticks(x + 0.01, df.columns)  # 적절하게 그리드 선에 맞춘다.
plt.grid()
plt.legend()


# #### 꺽은선 그래프

# In[251]:








# ### 원그래프

# In[267]:


import matplotlib.pyplot as plt 
import pandas as pd

index = ['1학년','2학년','3학년','4학년']
data  = [16,12,7,5]
df  = pd.DataFrame(val, index=index, columns=['도수'])

df['상대도수']  = [x/sum(data) for x in data]
df['백분율(%)'] = [x/sum(data)*100 for x in data]
df


# In[268]:


plt.pie(data, labels=data, autopct='%.1f%%', textprops={'size':12})
plt.legend(index, title='학년', loc="center right",  
           bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("학년별 동아리 회원수", size=15)
plt.show()


# --------

# ## 3.양적자료의 정리

# 숫자료 표현할 수 있는 자료의 정리

# ### [실습] 예제 3-8: 점도표 표현하기

# #### Q.숫자데이터 점도표로 표현하기

# In[280]:


import matplotlib.pyplot as plt 
import numpy as np

datas = [4.1, 4.2, 3.8, 4.2, 3.9, 4.2, 3.9, 4.1, 3.9, 4.3,
         3.9, 3.8, 3.8, 4.0, 4.3, 3.8, 3.9, 4.1, 4.1, 4.0]






# ### [실습] 도수분포표 만들기

# #### Q.핸드폰 사용시간을 계급수 K=5인 도수분포표를 만드시오

# - 올림: math.ceil()  # math 모듈내 함수
# - 내림: math.floor() # math 모듈내 함수
# - 반올림: round()    # 사사오입

# In[398]:


import math 
import numpy as np
import pandas as pd

data = [10,37,22,32,18,15,15,18,22,15,
       20,25,38,28,25,30,20,22,18,22,
       22,12,22,26,22,32,22,23,20,23,
       23,20,25,51,20,25,26,22,26,28,
       28,20,23,30,12,22,35,11,20,25]





# In[359]:


df


# ### 히스토그램

# 도수분포표로 작성한 자료를 시각적으로 쉽게 이해할 수 있도록 그린 그림

# #### Q.청소년 1주일 동안의 핸드폰 사용시간을 계급수 K=5인 히스토그램

# In[336]:


import matplotlib.pyplot as plt
import numpy as np

data = [10,37,22,32,18,15,15,18,22,15,
       20,25,38,28,25,30,20,22,18,22,
       22,12,22,26,22,32,22,23,20,23,
       23,20,25,51,20,25,26,22,26,28,
       28,20,23,30,12,22,35,11,20,25]

print(f'모집단: {len(data)}')
plt.hist(data)
plt.show()


# #### 계급 수 k=5 를 지정하여 히스토그램 그리기

# In[337]:


k = 5   #계급의 수

plt.hist(data, bins=k)
plt.grid()
plt.xlabel('핸드폰 사용 시간(hour)')
plt.ylabel('도수(frequency)')
plt.title('청소년의 1주일 동안의 핸드폰 사용 시간')
plt.show()


# #### 막대 그래프로 히스토그램 그리기

# In[338]:


x = df['계급값'].values[:5]
y = df['도수'].values[:5]
plt.bar(x,y, width=10)
plt.xticks(ticks=x, labels=x)
plt.xlabel('핸드폰 사용 시간(hour)')
plt.ylabel('도수(frequency)')
plt.title('청소년의 1주일 동안의 핸드폰 사용 시간')
plt.show()


# ### [실습] Q.3-9 예제 계급수 K=5인 히스토그램

# In[399]:


import math 
import numpy as np
import pandas as pd

data = [26,31,28,38,41,26,18,16,25,29,
       39,38,38,40,43,38,39,41,41,40,
       26,19,39,28,43,34,21,41,29,30,
       12,22,45,34,29,26,29,58,42,16,
       41,42,38,42,28,42,39,41,39,43]



# ### [실습] 예제 3-11 도수다각형 그리기

# 히스토그램에서 연속적인 막대의 상단 중심부를 선분으로 연결하여 다각형으로 표현한 그림

# #### 히스토그램 위에 도수다각형 그리기

# In[388]:


x = df['계급값'].values[:5]
y = df['도수'].values[:5]
print(y)

# 선그래프의 점의 시작과 끝 추가하기
z = np.zeros(1)
xa= np.array([x[0]-(x[1]-x[0])])
xb= np.array([x[-1]+(x[1]-x[0])])
x1= np.hstack([np.hstack([xa,x]),xb]) # 시작점 끝점 추가 
y1= np.hstack([np.hstack([z,y]),z])

plt.bar(x, y, width=10, edgecolor='w')
plt.plot(x1, y1, 'o', linestyle='solid', c='r')
plt.xticks(ticks=x1, labels=x1)
plt.grid()
plt.show()


# ### [실습]  줄기-잎 그림

# #### 방법1: stemgraphic 라이브러리 이용

# In[389]:


get_ipython().system('pip install stemgraphic  ')


# #### 예제 3-4 데이터 이용

# In[396]:


import stemgraphic

data = [10,37,22,32,18,15,15,18,22,15,
       20,25,38,28,25,30,20,22,18,22,
       22,12,22,26,22,32,22,23,20,23,
       23,20,25,51,20,25,26,22,26,28,
       28,20,23,30,12,22,35,11,20,25]

#stemgraphic.stem_graphic(data, scale=10)
stemgraphic.stem_graphic(data, scale=10, asc=False) 


# #### 방법2: pyplot.stem 이용

# In[391]:


stems = [i//10 for i in data]
stems.sort()
plt.stem(stems, data)
plt.xlabel('stem')
plt.ylabel('data')
plt.xticks(ticks=stems, label=stems)
plt.show()


# ### [실습] 예제 3-9 줄기-잎 그래프 그리기

# In[ ]:









# -------

# 끝
