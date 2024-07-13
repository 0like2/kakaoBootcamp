import pandas as pd

df = pd.read_csv('heart_2022_with_nans.csv')

# 데이터 몇줄 출력하여 구조 확인하기
print(df.head())

# 데이터 각 칼럼에 대한 정보 확인
print("데이터의 각 칼럼에 대한 정보 확인")
print(df.info())

# 피드백 1 : 코랩이랑 파이참이랑 콘솔에서 보이는 결과값이 다른데? 이걸 조정할 수 있는 방법이 있을까? 너무 적게 보여줘서요...

# 데이터 타입 확인
print("#"*10)
print("데이터 타입 확인")
print("Data Types :\n ", df.dtypes)

# 범주형 및 수치형 데이터 분리하여 분석
categorical_cols = df.select_dtypes(include=['object']).columns  # 실습자료에는 'category'포함이였으나 현 데이터에는 없음
print("#"*10)
print("범주형 데이터 분리하여 분석")
print(categorical_cols)

numerical_cols = df.select_dtypes(include=['float64']).columns
print("#"*10)
print("수치형 데이터 분리하여 분석")
print(numerical_cols)

# 결측치 확인
print("\nMissing Values:\n", df.isnull().sum())
print("#"*10)

# 각 범주형 변수의 유니크한 값과 빈도수
for col in categorical_cols:
    print(f"\nUnique Values in {col}:\n", df[col].value_counts())

# 수치형 데이터의 기초 통계
print("\nDescriptive Statistics for Numerical Data:\n", df[numerical_cols].describe())

# 왜도와 첨도 확인
print("\nSkewness of the data:\n", df[numerical_cols].skew())
print("\nKurtosis of the data:\n", df[numerical_cols].kurt())

'''
Skewness of the data:
 PhysicalHealthDays    2.179818
MentalHealthDays      2.123216
SleepHours            0.764602
HeightInMeters        0.028900
WeightInKilograms     1.075612
BMI                   1.387739
dtype: float64

결과 해석: 
양의 왜도: 분포의 꼬리가 오른쪽으로 길게 늘어진 경우(오른쪽 꼬리가 긴 분포).
음의 왜도: 분포의 꼬리가 왼쪽으로 길게 늘어진 경우(왼쪽 꼬리가 긴 분포).
왜도가 0: 대칭 분포(대부분의 경우 정규 분포를 의미).

numerical data 모두 skewness 값이 양수이고 그 크기는 PhysicalHealthDays, MentalHealthDays가 제일 크다. 그리고 SleepHours,HeightInmeters,WeightInKilograms,BMI순으로 커졌다. 
왜도가 0에 가까운 HeightInMeters,SleepHours가 대칭분포를 이루는 것을 알 수 있다. 이는, 키의 변화가 생기기 제한적이고 수면시간도 24시간내로 정해져있기때문에 특정 범위로 한정되기 때문이고 변화하기 어렵기 때문이다. 

'''

'''
Kurtosis of the data:
 PhysicalHealthDays    3.427589
MentalHealthDays      3.359229
SleepHours            8.741170
HeightInMeters        0.182299
WeightInKilograms     2.738972
BMI                   4.428387
dtype: float64


첨도 (Kurtosis):
	•	계산된 첨도 값이 3보다 크면, 분포가 두꺼운 꼬리를 가지며 극단치가 많음을 의미합니다 (Leptokurtic).
	•	3보다 작으면, 분포가 얇은 꼬리를 가지며 극단치가 적음을 의미합니다 (Platykurtic).
	•	3에 가까우면, 정규 분포에 가까움을 의미합니다 (Mesokurtic).

결과값: SleepHours가 8로 가장 높고 Height가 가장 낮다. 즉, 수면시간에 극단치가 많다는 것을 알 수 있다. 그다음으로 BMI가 극단치가 많다. 
'''
# 피어슨 & 스피어만 상관계수
correlation_matrix_pearson = df[numerical_cols].corr(method='pearson')
correlation_matrix_spearman = df[numerical_cols].corr(method='spearman')
print("\nPerson Correlation:\n", df[numerical_cols].corr(method='pearson'))
print("\nSpearman Correlation:\n", df[numerical_cols].corr(method='spearman'))



#특정 피어슨 결과
correlation_matrix_p_Sleeping_BMI = df['SleepHours'].corr(df['BMI'], method='pearson')
correlation_matrix_s_Sleeping_BMI = df['SleepHours'].corr(df['BMI'], method='spearman')
print("\nPerson Correlation with SleepHours/BMI:\n", correlation_matrix_p_Sleeping_BMI)
print("\nSpearman Correlation with SleepHours/BMI:\n", correlation_matrix_p_Sleeping_BMI)

'''
Person Correlation with SleepingHours/BMI:
 -0.05080531977201104

Spearman Correlation with SleepingHours/BMI:
 -0.05080531977201104
 
 둘다 모두 음의 상관관계가 나오므로 자는 시간이 증가함에 따라 BMI 수치는 줄어든다고 해석할 수 있다. 그 수치가 03 ~ 0.7 정도로 자는 시간이 증가함에 따라 BMI 수치는 살짝 줄어들게 된다. 
 
'''
# 특정 피어슨 결과 만들어내는 다른 방법
target_col = 'SleepHours'
correlations = df[numerical_cols].corr(method='pearson')[target_col]
print(f"Pearson Correlation between '{target_col}' and other columns:\n", correlations)

#correlations = df.corr(method='pearson')[target_col] 가 에러 뜬 이유는? 범주형 데이터도 포함시켜서!

# 만약에 자기자신의 상관계수 없앨려면?
correlations = correlations.drop(target_col)
from scipy import stats
# 데이터 분할
