import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

'''
막대그래프, 선그래프, 히스토그램, 파이차트, 산점도, 박스플롯을 구현해봅니다.
실습에 사용하는 데이터는 https://www.kaggle.com/datasets/nikhil7280/weather-type-classification 데이터입니다.
본인이 원하는 다른 데이터가 있다면 다른 데이터를 사용하셔도 좋습니다.
데이터에서 적절한 데이터를 선정하여 각 시각화 분석에 적용합니다.
시각화를 진행할 때는 차트 뿐만 아니라 범례, 제목, 레이블, 색상, 크기 등 여러가지 요소들을 직접 개발하시면서 데이터의 특성을 잘 이해해보시면 좋을 것 같습니다.
'''

df = pd.read_csv('weather_classification_data.csv')

"""
Data Types:
Temperature             float64
Humidity                  int64
Wind Speed              float64
Precipitation (%)       float64
Cloud Cover              object
Atmospheric Pressure    float64
UV Index                  int64
Season                   object
Visibility (km)         float64
Location                 object
Weather Type             object
dtype: object
###############
범주형 데이터:
Categorical_columns :
 Index(['Cloud Cover', 'Season', 'Location', 'Weather Type'], dtype='object')
###############
수치형 데이터:
Numerical_columns :
 Index(['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)',
       'Atmospheric Pressure', 'UV Index', 'Visibility (km)'],
      dtype='object')

"""
# 1. 막대그래프

# 데이터 준비
weather_count = df.groupby(['Location', 'Weather Type']).size().unstack()
print("weather_count:\n", weather_count)

#피드백 포인트1. size().unstack() VS count()
"""
`size().unstack()`과 `count()` 함수는 둘 다 데이터 그룹화 후 집계를 수행하지만, 그 목적과 사용법에는 몇 가지 중요한 차이점이 있습니다. 각각의 함수를 이해하기 위한 주요 차이점을 정리해 드리겠습니다:

### 1. **`size()` 함수**
- `size()` 함수는 각 그룹에 속하는 요소의 총 개수를 반환합니다. 이는 결측치를 포함한 모든 요소의 수를 계산합니다.
- `size()`는 보통 `groupby()`와 함께 사용되며, 그룹화된 각 그룹의 크기를 알고 싶을 때 유용합니다.
- 예를 들어, `data.groupby(['Location', 'Weather Type']).size()`는 각 위치와 날씨 유형 조합에 따른 데이터 포인트의 총 수를 계산합니다.

### 2. **`count()` 함수**
- `count()` 함수는 그룹 내에서 결측치가 아닌 값의 개수를 반환합니다. 즉, 결측치를 제외한 요소의 수를 계산합니다.
- 이는 특정 열에 대한 유효한 데이터의 수를 확인할 때 사용됩니다.
- 예를 들어, `data.groupby('Weather Type').count()`는 'Weather Type'별로 그룹화된 후 각 열에서 결측치가 아닌 데이터의 개수를 카운트합니다.

### `size().unstack()` vs. `count()`
- `size().unstack()` 조합은 그룹화 후 각 그룹의 크기를 행렬 형태로 재구조화하여 전체 데이터셋에 대한 간략한 개요를 제공합니다. 여기서 `unstack()`은 멀티인덱스를 가진 데이터를 재구조화하여 더 읽기 쉬운 형태로 변환합니다.
- 반면, `count()`는 각 그룹 내에서 결측치가 아닌 값의 개수만을 계산하고, 일반적으로 데이터의 완성도나 유효성을 검토할 때 사용됩니다.

따라서 사용 목적에 따라 적합한 함수를 선택하면 됩니다. 데이터의 전체 개수를 파악하고 싶다면 `size()`를, 유효한 데이터만을 카운트하고 싶다면 `count()`를 사용하는 것이 좋습니다.
"""
plt.figure(figsize=(10, 6))
bottom = np.zeros(len(weather_count))

for weather_type in weather_count.columns:
    plt.bar(weather_count.index, weather_count[weather_type], bottom=bottom, label = weather_type)
    bottom += weather_count[weather_type]
plt.title("Weather Type Distribution for each location")
plt.xlabel("Location")
plt.ylabel("Count")
plt.legend(title='Weather Type', bbox_to_anchor=(1.05, 1), loc='upper left')
'''
•	bbox_to_anchor=(1.05, 1): 범례의 위치를 그래프의 바깥으로 지정합니다. 여기서 (1.05, 1)은 범례가 그래프의 우측 상단 바깥에 위치하도록 합니다. 1.05는 x축 방향으로 그래프 경계에서 오른쪽으로 5% 떨어진 위치, 1은 y축 방향으로 그래프 상단 경계에서 위로 정렬된 위치를 의미합니다.
•	loc='upper left': 범례가 상자의 왼쪽 상단 근처에 위치하도록 설정합니다.
'''
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 피드백 2. weather_counts.plot(kind='bar', stacked=True) 와 plt.bar()의 차이가 뭐야? (여러값이 쌓여있는 막대바 그리기)
"""
`weather_counts.plot(kind='bar', stacked=True)`와 `plt.bar()`를 사용하여 막대 그래프를 그리는 데 있어 몇 가지 주요 차이점이 있습니다. 이 차이들은 특히 사용의 용이성과 커스터마이징 가능성에서 나타납니다.

### 1. **간결함과 용이성**

- **`weather_counts.plot(kind='bar', stacked=True)`**:
  - 이 구문은 pandas의 내장 함수를 사용하여 데이터프레임에서 직접 막대 그래프를 그립니다. `kind='bar'`는 막대 그래프 형태를, `stacked=True`는 막대를 쌓아서 표시하라는 것을 의미합니다.
  - 매우 간결하며, 복잡한 설정 없이도 즉시 그래프를 생성할 수 있습니다. 데이터프레임의 구조를 알고 있다면, 한 줄의 코드로 쉽게 시각화할 수 있습니다.
  - 이 방법은 pandas에 내장된 기능이기 때문에 데이터프레임과 직접 연동되어 데이터를 처리하고 시각화하는 과정이 매우 편리합니다.

- **`plt.bar()`**:
  - `plt.bar()`는 matplotlib의 기본 함수로, 더 세밀한 조정이 가능하지만 사용법이 조금 더 복잡합니다.
  - 각 막대의 위치, 높이, 너비, 색상 등을 상세히 지정할 수 있으며, 막대 위에 값 라벨을 추가하거나, 막대마다 다른 패턴을 적용하는 것과 같은 세밀한 커스터마이징이 가능합니다.
  - 막대를 쌓는 방식으로 그래프를 그리기 위해서는 누적 높이를 계산해야 하며, 이는 추가적인 코드 작성을 필요로 합니다.

### 2. **커스터마이징 가능성**

- `weather_counts.plot`은 pandas 스타일의 시각화로, 기본 설정 외에 커스터마이징 옵션이 제한적일 수 있습니다. 예를 들어, 막대의 특정 속성을 변경하기 위해서는 matplotlib의 기능을 추가로 사용해야 할 수 있습니다.
- `plt.bar()`는 matplotlib의 핵심 함수 중 하나로, 시각적 요소들을 더 세밀하게 조정할 수 있습니다. 막대의 정확한 위치 조정, 누적 방식 변경, 각 막대에 다른 패턴이나 텍스처 적용 등이 가능합니다.

### 결론

간단하고 빠르게 막대 그래프를 그리기 원한다면 `weather_counts.plot(kind='bar', stacked=True)`이 적합하며, 보다 복잡한 맞춤형 시각화가 필요한 경우 `plt.bar()`를 사용하는 것이 좋습니다. `plt.bar()`를 사용하면 초기 설정은 복잡할 수 있지만, 훨씬 더 유연한 시각화가 가능합니다.


plt.bar()를 사용하여 누적 막대 그래프를 구현하는 과정이 상대적으로 더 복잡하게 느껴질 수 있는 이유는, matplotlib를 이용한 막대 그래프 그리기에서 누적 표현을 수동으로 처리해야 하기 때문입니다. 여기에는 몇 가지 추가적인 계산과 설정이 필요합니다.

복잡성 증가의 이유

	1.	누적 높이 계산:
	•	plt.bar()를 사용할 때, 각 막대의 시작점을 직접 지정해야 합니다. 이는 이전 막대의 높이를 누적하여 현재 막대가 시작될 위치를 계산해야 함을 의미합니다. 이러한 누적 계산을 수동으로 수행해야 하며, 이는 코드를 더 길고 복잡하게 만듭니다.
	2.	반복 구조 필요:
	•	각 카테고리(날씨 유형)별로 막대를 그리기 위해 반복문을 사용해야 합니다. 각 반복에서 현재의 막대 높이를 이전 막대 높이에 추가하는 방식으로 그래프를 그립니다. 이 과정은 추가적인 코드 작성을 요구하며, 오류가 발생할 수 있는 여지를 만듭니다.
	3.	세밀한 매개변수 조정:
	•	plt.bar() 함수를 사용할 때 막대의 색상, 레이블, 너비, 간격 등을 개별적으로 설정할 수 있습니다. 이는 더 세밀한 시각적 조정을 가능하게 하지만, 동시에 그 설정을 관리해야 하는 복잡성을 추가합니다.

비교: Pandas plot() 사용 시 간소화

	•	반면, weather_counts.plot(kind='bar', stacked=True)와 같은 pandas 내장 그래프 함수를 사용하면, 이 모든 과정이 내부적으로 처리됩니다. 단 한 줄의 코드로 ‘누적 막대 그래프’를 생성할 수 있으며, 별도의 누적 높이 계산이나 반복 구조 없이 자동으로 누적 방식을 적용합니다.
	•	pandas의 그래프 함수는 데이터프레임의 구조를 활용하여 각 열을 자동으로 막대로 변환하고, 적절한 레이블과 색상을 적용하는 등의 작업을 간편하게 수행합니다.

결론적으로, plt.bar()를 사용하여 누적 막대 그래프를 만드는 것은 그 과정에서 보다 세밀한 컨트롤이 가능하지만, 그만큼 더 많은 설정과 코드가 필요하며, 이는 작업의 복잡성을 증가시키는 요소가 됩니다.
"""


# 피드백 3 : 여기서 나오는 시간은 시간이 아니라 단지 순서일 뿐이지 않을까? 시간에 대한 정보가 없을때 좋은 접근 방법은?
# 언제 사용할까? 주로 시간에 따른 데이터 변화를 관찰할때 좋음

# 인덱스를 시간 축으로 사용하면서 기온 변화 추이 그래프
plt.figure(figsize=(12, 6))
plt.plot(df['Temperature'], label='Temperature', color='red', linewidth=2)
plt.title('Temperature Change Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# 습도 변화 추이 그래프
plt.figure(figsize=(12, 6))
plt.plot(df['Humidity'], label='Humidity', color='blue', linewidth=2)
plt.title('Humidity Change Over Time')
plt.xlabel('Time')
plt.ylabel('Humidity (%)')
plt.legend()
plt.grid(True)
plt.show()


# 3. 히스토그램
plt.figure(figsize=(12, 6))
plt.hist(df['Visibility (km)'], bins = 20 , color='blue', alpha=0.5, label='Visibility (km)')
plt.title('Visibility (km)')
plt.xlabel('Visibility (km)')
plt.ylabel('frequency')
plt.grid(True)
plt.show()


# 4.파이차트

# 데이터 준비
weather_type_counts = df["Weather Type"].value_counts()

# 미리 설정하기
pie_colors = ["gray","blue","white","Red"]
pie_explode = (0,0,0,0.2)

# 데이터 시각화
plt.figure(figsize=(12, 6))
plt.pie(weather_type_counts, labels = weather_type_counts.index, colors = pie_colors, explode = pie_explode, autopct='%1.1f%%', startangle=140)
plt.title('Weather Type')
plt.axis('equal')
plt.show()

# 5. 산점도
plt.figure(figsize=(10, 6))
plt.scatter(df['Visibility (km)'], df['Wind Speed'], alpha=0.5, color='blue')
plt.title('Visibility (km) & Wind Speed')
plt.xlabel('Visibility (km)')
plt.ylabel('Wind Speed')
plt.grid(True)
plt.show()

# 6. 박스플롯
plt.figure(figsize=(12, 8))
sns.boxplot(x= "Weather Type", y="Visibility (kim)", data=df)
plt.title('Visibility (km) & Weather Type')
plt.xlabel('Visibility (km)')
plt.ylabel('Weather Type')
plt.grid(True)
plt.show()

# 7. 여러 시각화 차트 합쳐보기
