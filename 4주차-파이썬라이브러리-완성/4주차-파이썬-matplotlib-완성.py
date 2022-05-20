#!/usr/bin/env python
# coding: utf-8

# # 03.파이썬라이브러리-Matplotlib

# ### matplotlib 설치하기

# In[ ]:


get_ipython().system('pip install matplotlib    # 최초 한번만 실행하기')


# -----------

# ### matplotlib 사용하기

# In[ ]:


import matplotlib.pyplot as plt


# ### matplotlib 사용 예:

# In[10]:


import matplotlib
import matplotlib.pyplot as plt

# 한글출력 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'   

# 그래프 크기 설정
plt.rcParams['figure.figsize'] = (10, 5)  # (가로,세로) 인치 단위
#plt.figure(figsize=(10, 5))

# X축, Y축 데이터
X = list(range(2005,2015,1))
Y = [15,7,2,10,8,5,14,9,18,8]


plt.plot(X, Y, c='r')     #선 그래프
plt.grid(True)
plt.title('년도별 그래프')
plt.show()


# ### matplotlib  선그래프 나타내기

# In[32]:


import matplotlib.pyplot as plt
import pandas as pd

# 한글출력 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # '맑은 고딕'으로 설정 
plt.rcParams['axes.unicode_minus'] = False     # 그래프에서 (-)숫자표시

# 그래프 크기 설정
plt.rcParams['figure.figsize'] = (10, 5) # 그래프(figure)의 크기, (가로,세로) 인치 단위
plt.rcParams['lines.linewidth'] = 3      # 선 두께
plt.rcParams['axes.grid'] = True 

data    = [[15,7,2,10,8,5,14,9,18,8]]  
columns = list(range(2005,2015,1))
index   = ['횟수']

df = pd.DataFrame(data, columns=columns, index=index)
df.columns.name = '연도'

x = df.columns    #x = df.columns
y = df.values[0]  #y = df.values[0]

plt.plot(x, y, 'o', linestyle='dashed', c='b')
plt.title('년도별 그래프')
plt.show()


# ### [실습] : 2개 선그래프 나타내기

# In[88]:


import matplotlib.pyplot as plt

# 한글출력 설정
plt.rcParams['font.family'] = 'Malgun Gothic'   

# 그래프 크기 설정
plt.rcParams['figure.figsize'] = (10, 5)  # (가로,세로) 인치 단위

# X축, Y축 데이터
X1 = list(range(2005,2015,1))
X2 = list(range(2008,2018,1))
Y = [15,7,2,10,8,5,14,9,18,8]

plt.plot(X1, Y, 'o', linestyle='solid', c='c', label='X1')     #선 그래프
plt.plot(X2, Y, 'X', linestyle='solid', c='m', label='X2')     #선 그래프
plt.title('년도별 그래프')
plt.grid(False)
plt.legend()
plt.show()


# ### 선 그래프 : numpy 데이터

# In[37]:


import matplotlib.pyplot as plt
import numpy as np

a = np.array([10,14,19,20,25])
plt.plot(a)

a2 = np.arange(20)
plt.plot(a2)

plt.show() 


# ### 2차 방정식의 그래프 : f(x) = a*x^2 + b, 포물선 방정식

# In[40]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-4.5,5,0.5)
y = 2*x**2
plt.plot(x, y)

plt.plot(x, 4*x)
plt.plot(x,-3*x)
plt.show()


# -------

# ### 막대 그래프 나타내기

# In[107]:


import matplotlib.pyplot as plt

# 한글출력 설정
plt.rcParams['font.family'] = 'Malgun Gothic'   

# 그래프 크기 설정
plt.rcParams['figure.figsize'] = (10, 5)  # (가로,세로) 인치 단위

# X축, Y축 데이터
X = list(range(2005,2015,1))
Y = [15,7,2,10,8,5,14,9,18,8]

plt.bar(X, Y)     # 막대 그래프
plt.grid(False)
plt.title('년도별 그래프')
plt.show()


# In[46]:


plt.barh(X, Y)     # 막대 그래프
plt.grid(False)
plt.title('년도별 그래프')
plt.show()


# In[49]:


#그래프 색상 지정
colors = ['r','g','b','c','m','y','k','w']  # 기본색상  # Hexa코드 or CSS컬러사용
plt.bar(X, Y,color=colors)
plt.show()


# In[66]:


import matplotlib.pyplot as plt

# 한글출력 설정
plt.rcParams['font.family'] = 'Malgun Gothic'   

# 그래프 크기 설정
plt.rcParams['figure.figsize'] = (10, 5)  # (가로,세로) 인치 단위

# X축, Y축 데이터
X1 = list(range(2005,2015,1))
X2 = [x+3 for x in X1]
Y = [15,7,2,10,8,5,14,9,18,8]

plt.bar(X1, Y)     # 막대 그래프
plt.bar(X2, Y)     # 막대 그래프
plt.title('년도별 그래프')
plt.grid(False)
plt.show()


# ### 2개 막대 그래프

# In[83]:


import matplotlib.pyplot as plt
import numpy as np

# 한글출력 설정
plt.rcParams['font.family'] = 'Malgun Gothic'   

# 그래프 크기 설정
plt.rcParams['figure.figsize'] = (10, 5)  

# X축, Y축 데이터
X = list(range(2005,2015,1))
Y1= [15,7,2,10,8,5,14,9,18,8]
Y2= [y+1 for y in Y1]
lable = X
X = np.arange(len(X))

plt.bar(X-0.2, Y1, width=0.4, label='Y1')     # 막대 그래프
plt.bar(X+0.2, Y2, width=0.4, label='Y2')     # 막대 그래프
plt.xticks(X, lable)

plt.title('년도별 그래프')
plt.grid(False)
plt.legend()
plt.show()


# In[92]:


# X축, Y축 데이터
X = list(range(2005,2015,1))
Y1= [15,7,2,10,8,5,14,9,18,8]
Y2= [y+1 for y in Y1]

plt.plot(X, Y1, 'o', linestyle='dashed', c='r', label='Y1') # 선 그래프
plt.bar(X, Y2, width=0.4, label='Y2')     # 막대 그래프
plt.xticks(X, lable)

plt.title('년도별 그래프')
plt.grid(False)
plt.legend()
plt.show()


# -------

# ### 원 그래프

# In[111]:


import matplotlib.pyplot as plt 
import pandas as pd

idx = ['1학년','2학년','3학년','4학년']
val = [16,12,7,5]
per = [val[x]/sum(val) *100 for x in range(len(val))]

plt.pie(per, autopct='%1.1f%%',
              textprops=dict(color="w"))
plt.title("학년별 동아리 회원수")
plt.legend(idx, title='학년', loc="center right",
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()


# ----------

# # 04. 실습:공공데이터 분석

# ###  공공데이터 수집
# 
# - 기상청 사이트에서 날씨 데이터를 다운로드한다.
# - 사이트에 가입하고 로그인해야 다운로드할 수 있다.
# - https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36
# - csv파일 다운로드해서 사용
# - OBS_ASOS_MNH_~.csv 파일 이름을 'weather_data.csv'로 변경해서 사용하기
# 

# ---------

# ## 데이터 가공 및 시각화하기

# ### 4-1. 데이터 가공하기

# #### 1. 다운로드된 데이터 불러오기 

# In[2]:


import pandas as pd

# OBS_ASOS_MNH_~.csv 파일명 데이터를 'weather_data.csv'로 변경하기 
#file = 'C:/python/gachon/2022_1_Statistics/data/weather_data.csv' 
file = 'data/weather_data.csv' 
df = pd.read_csv(file, encoding='CP949', engine='python')
print(type(df))
df


# #### #데이터 요약

# In[3]:


df.describe()


# #### 2. 컬럼명 변경하기

# In[4]:


df.columns
df.columns = ['지점', '지점명', '일시', '평균기온', '최고기온', '최저기온', '평균현지기압',
       '평균상대습도', '월합강수량', '평균풍속','일조시간', '최심적설']
df


# #### 3. 컬럼 추가하기
# DataFrame.insert(추가하고싶은위치,컬럼명,값, allow_duplicate=False)

# In[5]:


df.insert(3,'신규',df['지점'])
df


# In[6]:


# 년월 추가
df.insert(3,'년도',df['일시'].str[0:4])
df


# #### 4. 컬럼 삭제

# In[7]:


df.drop('신규', axis=1, inplace=True)
df


# #### 5. 특정 조건 검색

# In[8]:


print( len( set(df['지점명']) ) )
print( set( df['지점명'] ) )


# #### #지점명 '제주' 데이터

# In[9]:


df.query(" 지점명 == '제주' ")

df.iloc[ df.index[ df['지점명'] == '제주' ] ]


# #### #최초 측정월, 마지막 측정월

# In[10]:


# 측정월 최소, 최대 
print ( min(df['일시']) )
print ( df['일시'].min() )
print ( max(df['일시']) )
print ( df['일시'].max() )


# In[11]:


# 측정 최초월('1904-04')의 데이터
df.query(" 일시 == '1904-04' ")

df.query(f" 일시 == '{df['일시'].min()}' ")

df.iloc[ df.index[ df['일시'] == df['일시'].min() ] ]


# ### [실습]: 가장 더웠던 날짜와 온도는?

# In[12]:


field = '최고기온'
wmax = df[field].max()
cond = f" {field} >= {wmax}"
print(wmax, cond)
df.query(cond)
df.query("최고기온 >= 41.0")

# print(df[field].max(), f" {field} >= {df[field].max()}")
# df.query(f" {field} >= { df[field].max() }")


# ### [실습]: 가장 추웠던 날짜와 온도는?

# In[13]:


field = '최저기온'
# wmin = df[field].min()
# cond = f" {field} <= {wmin}"
# df.query(cond)
# df.query("최저기온 <= -32.6")

print(df[field].min(), f" {field} >= {df[field].min()}")
df.query(f" {field} <= { df[field].min() }")


# ###  [실습] : '제주'에서 가장 더웠던 날짜와 기온?

# In[14]:


wmax = df.query(" 지점명 == '제주' ")['최고기온'].max()
df.query(f" 지점명 == '제주' and 최고기온 >= { wmax }")


# ###  [실습] : '제주'에서 가장 추웠던 날짜와 기온은?

# In[15]:


wmin = df.query(" 지점명 == '제주' ")['최저기온'].min() 
df.query(f" 지점명 == '제주' and 최저기온 <= { wmin  } ")


# --------

# #### #그룹핑: 지점별 평균 기온 

# In[16]:


df.groupby(['지점명'], as_index=False).mean()


# In[17]:


df.groupby(['지점명'], as_index=False).count()


# ### [실습] : 지점별 평균기온 

# #### 방법1: groupby 이용

# In[18]:


# grouped =df['평균기온'].groupby(df['지점명'])
# grouped.mean()

df['평균기온'].groupby(df['지점명']).mean()


# ####  방법2: agg() 사용

# #### 컬럼 1개: 지점명별 그룹

# In[19]:


df.groupby('지점명')['평균기온'].agg(**{'평균기온':'mean'})

#df.groupby('지점명')['평균기온'].agg(**{'mean_temp':'mean'}).reset_index()


# #### 컬럼 2개: 지점명, 년도별 그룹

# In[20]:


df.groupby(['지점명','년도'])['평균기온'].agg(**{'평균기온':'mean'})
#df.groupby(['지점명','년도'])['평균기온'].agg(**{'평균기온':'mean'}).reset_index()


# ### [실습] : 지점별 최고기온?

# In[21]:


#df.groupby('지점명')['최고기온'].agg(**{'최고기온':'max'}).reset_index()
#df.groupby('지점명')['최고기온'].agg(**{'최고기온':'max'})

df2 = df.groupby('지점명')['최고기온'].agg(**{'최고기온':'max'})


# ### [실습] : 지점별 년도별 최대강수량?

# In[22]:


df.groupby(['지점명','년도'])['최고기온'].agg(**{'일최다강수량':'max'})


# #### #한 행씩 불러오기
# !!![주의]!!! 데이터가 많을 경우 실행하지 않도록

# In[23]:


# 한 행씩 불러오기  ---->
for i, row in df.iterrows():
    #print(i, row)
    print(i, row['지점명'], row['최고기온'])


# #### 6.가공된 파일 신규 저장하기

# In[25]:


# 가공된 데이터 신규 파일로 저장하기
#file = 'c:/python/gachon/2022_1_statistics/data/weather_data2.csv'    
file = 'data/weather_data2.csv' 
df.to_csv(file, encoding='cp949', mode='w', index=True)


file = 'data/weather_group_max.csv'    
df2.to_csv(file, encoding='cp949', mode='w', index=True)


# --------------------------

# ### 4-2. 데이터 시각화하기: 그래프

# ### [실습] : 지점별 평균기온 그래프

# In[26]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 한글출력 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 가져오기
file = 'data/weather_data2.csv'   
df = pd.read_csv(file, encoding='CP949', engine='python')


# In[27]:


# 2. 지점별 평균기온 데이터 추출하기: reset_index()
data = df.groupby('지점명')['평균기온'].agg(**{'평균기온':'mean'})
# print(data)

# 3. 그래프로 시각화하기 
plt.figure(figsize=(15,5))     # 그래프 크기
plt.title('지점명 평균기온 통계') #제목
plt.legend(['평균기온'])  # 범례,기본 위치 : loc='upper left'
plt.xlabel('지점')        # x축 레이블
plt.ylabel('평균기온')    # y축 레이블
plt.xticks(rotation=45)  # x축 레이블 기울이기
plt.grid()               # 격자 표시
plt.plot(data, 'o', linestyle='solid', c='r')  
plt.show()


# ### [실습] : '제주' 지점의 년도별 평균기온을  선 그래프로 나타내기

# In[28]:


# 2. '제주'지점 년도별 평균기온 데이터 추출하기: reset_index()
# data = df.query(" 지점명 == '제주' ")
# data = data.groupby('년도')['평균기온'].agg(**{'평균기온':'mean'}).reset_index()

data = df.groupby(['지점명','년도'])['평균기온'].agg(**{'평균기온':'mean'}).reset_index()
data = data.query(" 지점명 == '제주' ")

X = data['년도']
Y = data['평균기온']

# 3. 그래프로 시각화하기 
plt.figure(figsize=(15,5))
plt.title("'제주'지점 년도별 평균기온 통계", fontsize=15) #제목
plt.xlabel('년도')        # x축 레이블
plt.ylabel('평균기온')    # y축 레이블             
plt.legend(['평균기온'])  #범례,기본 위치 : loc='upper left'
plt.xticks(rotation=45)  # x축 레이블 기울이기
plt.grid()               # 격자 표시
plt.plot(X, Y, 'o', linestyle='solid', c='r')  
plt.show()


# ### [실습]: 제주의 평균기온, 최고기온, 최저기온 통계를 그래프로 나타내기

# In[29]:


# 2. '제주'지점 년도별 평균기온, 최고기온, 최저기온 데이터 추출하기: reset_index()
data = df.groupby(['지점명','년도'])['평균기온'].agg(**{'평균기온':'mean'}).reset_index()
data = data.query(" 지점명 == '제주' ")
X  = data['년도']
Y1 = data['평균기온']

data = df.groupby(['지점명','년도'])['최고기온'].agg(**{'최고기온':'max'}).reset_index()
data = data.query(" 지점명 == '제주' ")
Y2 = data['최고기온']
data = df.groupby(['지점명','년도'])['최저기온'].agg(**{'최저기온':'min'}).reset_index()
data = data.query(" 지점명 == '제주' ")
Y3 = data['최저기온']

# 3. 그래프로 시각화하기 
plt.figure(figsize=(20,5))
plt.title('제주 기온 통계', fontsize=15) #제목
plt.legend(['평균기온','최고기온','최저기온'])  #범례,기본 위치 : loc='upper left'
plt.xlabel('년도')        # x축 레이블
plt.ylabel('기온')    # y축 레이블 
plt.xticks(rotation=45)  # x축 레이블 기울이기
plt.grid()               # 격자 표시

plt.plot(X, Y1, 'o', linestyle='solid', c='r')  # 선그래프
plt.plot(X, Y2, 'o', linestyle='solid', c='g')  # 선그래프
plt.plot(X, Y3, 'o', linestyle='solid', c='b')  # 선그래프
plt.show()


# ### [실습]: 제주의 최고기온(선그래프),평균기온(막대그래프)을 혼합 그래프로 나타내기

# In[30]:


# 3. 그래프로 시각화하기 
plt.figure(figsize=(20,5))
plt.title('제주 기온 통계', fontsize=15)         #제목
plt.legend(['최고기온','평균기온']) #범례,기본 위치 : loc='upper left'
plt.xlabel('년도')                 # x축 레이블
plt.ylabel('기온')                 # y축 레이블  
plt.xticks(rotation=45)           # x축 레이블 기울이기
plt.grid()                        # 격자 표시

plt.bar(X, Y1)  # 막대 그래프
plt.plot(X, Y2, 'o', linestyle='solid', c='r')  # 선그래프
plt.show()


# In[34]:


df


# ### [실습] : 2중 축 그래프 그리기 :  평균상대습도 & 월합강수량 

# In[36]:


# 2.'제주'지점의 평균상대습도 & 일최다강수량 센터수
data = df.groupby(['지점명','년도'])['평균상대습도'].agg(**{'평균상대습도':'mean'}).reset_index()
data = data.query(" 지점명 == '제주' ")
X  = data['년도']
Y1 = data['평균상대습도']

data = df.groupby(['지점명','년도'])['월합강수량'].agg(**{'월합강수량':'mean'}).reset_index()
data = data.query(" 지점명 == '제주' ")
Y2 = data['월합강수량']
print(Y2)


# 3. 그래프로 시각화하기 
plt.rcParams['figure.figsize'] = (10, 5)   # 그래프 크기

fig, ax1 = plt.subplots()        
plt.suptitle('평균상대습도 & 월합강수량',fontsize=15)

ax1.plot(X, Y1, color='green')     # 평균상대습도 (왼쪽)
ax2 = ax1.twinx()
ax2.plot(X, Y2, color='deeppink')  # 월합강수량 (오른쪽)

fig.autofmt_xdate(rotation=45)  #X축 레이블 기울이기

plt.show()


# -------

# 끝!
