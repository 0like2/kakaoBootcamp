'''
1. 고급 차트 유형에서 제시한 5개 시각화 방법 사용해보기
● 히트맵
● 트리맵
● 버블차트
● 레이더 차트
● 생키 다이어그램
2. 시각화한 차트를 인터렉티브하게 만들어보기
3. 지리 정보 데이터 시각화 해보기
● geopandas
● folium
'''


# 1. 고급 차트 유형 시각화 방법 사용해보기

# 피드백 1 : 주로 데이터는 어디서 얻어오는가? 캐글, 공공데이터포털,

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# https://e-jhis.org/journal/view.php?viewtype=pubreader&number=837 참고 자료
# https://e-medis.nemc.or.kr/portal/theme/dataVisualPage.do 의료데이터 시각화 예시

# 데이터 셋 준비
df = pd.read_csv('Healthcare-Diabetes.csv')
print(df.head())

print("데이터 칼럼 정보 확인 :")
print(df.info())

print("데이터 타입 확인")
print("Data Types:\n", df.dtypes)

"""
Data Types:
 Id                            int64
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object
"""

print("Missing Value: \n", df.isnull().sum())

print(df.describe())

# age 카테고리 만들기
# Define age categories and labels
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = ['0-10', '10대', '20대', '30대', '40대', '50대', '60대', '70대', '80대'] # max age = 81.0

# Create a new column 'AgeCategory' based on the defined bins and labels
# df['AgeCategory'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

num_bins = 10
# Create a new column 'AgeCategory_qcut' using qcut to define bins with approximately equal frequency
df['AgeCategory_qcut'] = pd.qcut(df['Age'], num_bins, labels=[f'Bin{i}' for i in range(1, num_bins+1)])

# 피드백2 : 데이터를 범주형으로 만드는 가장 간단한 방법은??

df.head()

# 히트맵
bmi_values = df['BMI'].values
bmi_matrix = bmi_values.reshape((int(len(bmi_values)), -1))

plt.figure(figsize=(10, 8))
sns.heatmap(bmi_matrix, annot=True, cmap="coolwarm")
plt.title("BMI 히트맵")
plt.show()


# 트리맵
# import plotly.express as px


# 버블차트
x = df["Age"]
y = df["Glucose"]
sizes = df["BMI"] * 10 #숫자를 고려하여 임의로 조절

plt.figure(figsize=(14, 10))
plt.scatter(x,y, s=sizes, alpha=0.5, cmap="viridis")
plt.title("BMI bubble chart(AGE VS GLUCOSE)")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.colorbar(label="BMI")
plt.show()

# 레이더 차트
select_df = df.loc[0, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
num_var=len(select_df)

angles = np.linspace(0, 2*np.pi, num=num_var, endpoint=False).tolist()
values = select_df.values.flatten().tolist()
values += values[:1]
angles += angles[:1]

# 생키 다이어그램
import plotly.graph_objects as go

# 데이터 정의
source = df["Pregnancies"].astype(str)
target = df["Outcome"].astype(str)

sankey_data = pd.DataFrame({'source': source, 'target': target})
sankey_counts = sankey_data.value_counts().reset_index(name='value')

all_labels = list(set(sankey_counts['source'].tolist() + sankey_counts['target'].tolist()))

sankey_counts['source_id'] = sankey_counts['source'].apply(lambda x: all_labels.index(x))
sankey_counts['target_id'] = sankey_counts['target'].apply(lambda x: all_labels.index(x))


# Define the nodes and links
nodes = all_labels
links = {
    "source": sankey_counts['source_id'].tolist(),
    "target": sankey_counts['target_id'].tolist(),
    "value": sankey_counts['value'].tolist()
}

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes
    ),
    link=dict(
        source=links["source"],
        target=links["target"],
        value=links["value"]
    )
))

fig.update_layout(title_text="Pregnancies to Outcome Sankey Diagram", font_size=10)
fig.show()
