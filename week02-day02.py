import pandas as pd

df = pd.read_csv("weather_classification_data.csv")

# 데이터 몇줄 간단하게 확인
print("#"*15)
print(df.head())

# 데이터 칼럼 정보 확인
print("#"*15)
print("데이터 칼럼 정보 확인 :")
print(df.info())

# Data type 확인
print("#"*15)
print("데이터 타입 확인")
print("Data Types:\n", df.dtypes)

# 데이터 범주/수치형 데이터 분류 및 분석
categorical_num = df.select_dtypes(include=['object']).columns
print("#"*15)
print("범주형 데이터:")
print("Categorical_columns :\n",categorical_num)

numerical_num = df.select_dtypes(include=['float64','int64']).columns
print("#"*15)
print('수치형 데이터:')
print("Numerical_columns :\n",numerical_num)

# 결측치 확인
print("Missing Values:\n",df.isnull().sum())
"""
Missing Values:
Temperature            0
Humidity                0
Wind Speed              0
Precipitation (%)       0
Cloud Cover             0
Atmospheric Pressure    0
UV Index                0
Season                  0
Visibility (km)         0
Location                0
Weather Type            0
dtype: int64
"""


# 범주형 데이터 확인 및 수치형 데이터 기초 통계
for col in categorical_num:
    print(f"Unique Value in {col}:\n", df[col].value_counts())

print("Descriptive Statistics for numerical Data:\n",df[numerical_num].describe())

"""
Unique Value in Cloud Cover:
 Cloud Cover
overcast         6090
partly cloudy    4560
clear            2139
cloudy            411
Name: count, dtype: int64
Unique Value in Season:
 Season
Winter    5610
Spring    2598
Autumn    2500
Summer    2492
Name: count, dtype: int64
Unique Value in Location:
 Location
inland      4816
mountain    4813
coastal     3571
Name: count, dtype: int64
Unique Value in Weather Type:
 Weather Type
Rainy     3300
Cloudy    3300
Sunny     3300
Snowy     3300
Name: count, dtype: int64
"""

"""
Descriptive Statistics for numerical Data:
         Temperature      Humidity  ...      UV Index  Visibility (km)
count  13200.000000  13200.000000  ...  13200.000000     13200.000000
mean      19.127576     68.710833  ...      4.005758         5.462917
std       17.386327     20.194248  ...      3.856600         3.371499
min      -25.000000     20.000000  ...      0.000000         0.000000
25%        4.000000     57.000000  ...      1.000000         3.000000
50%       21.000000     70.000000  ...      3.000000         5.000000
75%       31.000000     84.000000  ...      7.000000         7.500000
max      109.000000    109.000000  ...     14.000000        20.000000
"""

print("Skewness of the Data:\n", df[numerical_num].skew())
print("kurtosis of the Data:\n", df[numerical_num].kurt())

"""
Skewness of the Data:
 Temperature             0.221741
Humidity               -0.401614
Wind Speed              1.360263
Precipitation (%)      -0.152457
Atmospheric Pressure   -0.293899
UV Index                0.900010
Visibility (km)         1.233275
dtype: float64
kurtosis of the Data:
 Temperature              0.586051
Humidity                -0.338366
Wind Speed               3.255194
Precipitation (%)       -1.354039
Atmospheric Pressure    12.778071
UV Index                -0.362166
Visibility (km)          2.517275
"""

'''
Wind Speed              1.360263
UV Index                0.900010
skewed Data 전처리 필요 -> log 함수로 만들기

Wind Speed               3.255194
 Atmospheric Pressure    12.778071
전처리 필요!

'''
# skewed data 전처리
import numpy as np

df['Wind Speed'] = np.log(df['Wind Speed'] + 1)  # 로그 변환 (0으로 나누는 오류 방지 위해 1 추가)
df['UV Index'] = np.log(df['UV Index'] + 1)      # 로그 변환 (0으로 나누는 오류 방지 위해 1 추가)

# 피드백1: 걍 standarzation 하면 되는거 아닌가?(어떤경우에 굳이 로그 변환을 해주어야 할까?)


print("Pearson Correlation:\n",df[numerical_num].corr(method="pearson"))


from scipy import stats

#계졀별 visibility 차이 분석

summer_visibility = df[df["Season"] == "Summer"]["Visibility (km)"]
winter_visibility = df[df["Season"] == "Winter"]["Visibility (km)"]

t_stat,p_val = stats.ttest_ind(summer_visibility,winter_visibility)
print(f"T-statistic : {t_stat}, P-value : {p_val}")

# UV index와 visibility 관계 분석
correlation_visibility_UV = df['Visibility (km)'].corr(df['UV Index'], method='pearson')
print("\nPerson Correlation with Visibility/UV:\n", correlation_visibility_UV)

pearson_corr,p_val = stats.pearsonr(df['Visibility (km)'], df['UV Index'])
print(f"P-value: {p_val}")

# p-value까지 구할거면 pearson_corr,p_Val = stat. pearsonr 사용

'''
T-statistic : 23.404802226814734, P-value : 2.7902128045161302e-117
t-통계량이 23.4로 매우 크다. 즉, 두 그릅의 평균 차이가 크다는 것이고 p-value의 값이 0.05보다 작으므로 여름과 겨울의 가시거리 차이가 통계적으로 유의미하다. 

Person Correlation with Visibility/UV:
 0.4109204947364387
P-value: 0.0
높은 UV index에서는 visibility도 증가한다. 
상식적으로도, 높은 uv index는 맑은 날씨이며, 이에 따라 가시성이 좋아진다라고 할 수 있다. 
'''
#피드백2 : p-value가 0.0이라 제대로 계산한건지 의심스러운데 확인할 방법이 궁금하다!

# 어떤 인사이트를 얻을 수 있까? 어떤 서비스를 개발할 수 있을까?
'''
계절별로는 visibility가 달라짐을 알 수 있다. 또한, uv index가 커짐에 따라 visibility가 좋아진다.
이를 이용하여 visibility가 좋아지는 관련 요소를 정리하여 계절, UV index, humid 등을 이용하여 visiblilty가 좋은 정도를 예측한다. 
일반적으로 visibility가 좋은날을 골라 사진을 찍으러 나간다. 특히, 사진작가들의 경우, 야외 촬영시 이를 고려한다. 이를 위해 사진찍으러 나가기 좋은날인지 팝 알림을 주는 서비스를 개발한다. 
'''
