'''실습!
kaggle의 데이터를 선정해 데이터 전처리 파이프라인 구축해보기
 ● 전처리가 쉬운 데이터를 선택하기 보다 본인이 흥미있는 분야 데이터를 선정 권장
 ● 전처리 시나리오
 ● 전처리 코드
 ● 전처리 전후 데이터
 ● 전처리방법 적용 이유
 ● 기타
 '''


''' 피드백 사항
1. 잔처리 시나리오란? 
2. Bloodpressure 
'''

# ● 전처리가 쉬운 데이터를 선택하기 보다 본인이 흥미있는 분야 데이터를 선정 권장 ->Healthcare 관련 분야로 선택
# Healthcare Diabetes Datasets : https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes

# ● 전처리 시나리오
# 1. 데이터 로드 -> 2.데이터 확인 -> 3. 결측치 확인 및 처리 -> 4. 데이터 변환 및 인코딩 -> 5. 특성 스케일링 -> 6. 불필요한 열 제거

# ● 전처리 코드
# 1. 데이터 로드
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Healthcare-Diabetes.csv')

# 2. 데이터 확인
print(df.head())
print(df.info())
print(df.describe())
columns = df.columns
print(columns)


'''
   Id  Pregnancies  Glucose  ...  DiabetesPedigreeFunction  Age  Outcome
0   1            6      148  ...                     0.627   50        1
1   2            1       85  ...                     0.351   31        0
2   3            8      183  ...                     0.672   32        1
3   4            1       89  ...                     0.167   21        0
4   5            0      137  ...                     2.288   33        1

[5 rows x 10 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2768 entries, 0 to 2767
Data columns (total 10 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Id                        2768 non-null   int64  
 1   Pregnancies               2768 non-null   int64  
 2   Glucose                   2768 non-null   int64  
 3   BloodPressure             2768 non-null   int64  
 4   SkinThickness             2768 non-null   int64  
 5   Insulin                   2768 non-null   int64  
 6   BMI                       2768 non-null   float64
 7   DiabetesPedigreeFunction  2768 non-null   float64
 8   Age                       2768 non-null   int64  
 9   Outcome                   2768 non-null   int64  
dtypes: float64(2), int64(8)
memory usage: 216.4 KB
None
                Id  Pregnancies  ...          Age      Outcome
count  2768.000000  2768.000000  ...  2768.000000  2768.000000
mean   1384.500000     3.742775  ...    33.132225     0.343931
std     799.197097     3.323801  ...    11.777230     0.475104
min       1.000000     0.000000  ...    21.000000     0.000000
25%     692.750000     1.000000  ...    24.000000     0.000000
50%    1384.500000     3.000000  ...    29.000000     0.000000
75%    2076.250000     6.000000  ...    40.000000     1.000000
max    2768.000000    17.000000  ...    81.000000     1.000000

'''
'''
Index(['Id', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')
'''

# 3. 결측치 확인 및 처리
print(df.isnull().sum())

'''
Id                          0
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
'''

# 4. 데이터 변환 및 인코딩
# Gemder값 변형
if "Gender" in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# BloodPressure 범위구간에 따라 High= 3, normal = 2, low =1


# 5. 특성 스케일링
# 어떤 열에 스케일링을 진행할지??  -> 1. 수치형 범주 2. 분포 확인 3. 변수 스케일 확인
import matplotlib.pyplot as plt

'''
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
# numeric_cols = df.select_dtypes(include=['int64', 'float64']) 와 numeric_cols = df.select_dtypes(include='numbers') 차이는?
print(numeric_cols)

df[numeric_cols].hist(bins=15, figsize=(15,10))
plt.show()

print(df[numeric_cols].describe())
'''

# 수치형 변수만 선택
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Numeric columns:", numeric_cols)

# 수치형 변수의 히스토그램 그리기
try:
    df[numeric_cols].hist(bins=15, figsize=(15, 10))
    plt.show()
except Exception as e:
    pㅋ노 rint(f"An error occurred: {e}")

# 각 변수의 값 범위 확인
print(df[numeric_cols].describe())

'''
Numeric columns: Index(['Id', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')
                Id  Pregnancies  ...          Age      Outcome
count  2768.000000  2768.000000  ...  2768.000000  2768.000000
mean   1384.500000     3.742775  ...    33.132225     0.343931
std     799.197097     3.323801  ...    11.777230     0.475104
min       1.000000     0.000000  ...    21.000000     0.000000
25%     692.750000     1.000000  ...    24.000000     0.000000
50%    1384.500000     3.000000  ...    29.0000술0     0.000000
75%    2076.250000     6.000000  ...    40.000000     1.000000
max    2768.000000    17.000000  ...    81.000000     1.000000

'''
# pregnancies & skin thickness 정규화
cols_to_normalize = ['Pregnancies','SkinThickness']
scaler = StandardScaler()
df[cols_to_normalize] = scaler.fit_transform(df[numeric_cols])
print(df[cols_to_normalize].describe())


# 6. 불필요한 열 제거
df = df.drop(columns=['DiabetesPedigreeFunction'])

# ● 전처리방법 적용 이유
# 각과정에 기