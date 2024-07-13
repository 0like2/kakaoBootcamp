'''
과제 1: 회귀 분석

목표: 특정 변수에 대한 선형 회귀 모델을 구축하고 평가합니다.
데이터셋: California Housing dataset을 사용합니다.
내용:
1.데이터를 불러오고, 독립 변수(X)와 종속 변수(y)를 설정합니다.
2.훈련 데이터와 테스트 데이터로 분할합니다.
3.선형 회귀 모델을 학습시키고, 테스트 데이터에 대해 예측합니다.
4.예측 결과를 평가하고, MSE와 R^2 점수를 계산합니다.
5.실제 값과 예측 값을 시각화합니다.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, mean_squared_error, r2_score

from sklearn.datasets import fetch_california_housing
from sklearn.naive_bayes import GaussianNB

# 데이터 로드
california = fetch_california_housing()

# 데이터 확인? -> 새로 알게 된 부분! -> 피드백 0: 이부분은 사실 챗 gpt 쓴건데 원래라면 데이터 파악을 어떻게 하는게 맞았을까?
print("Data shape:", california.data.shape)
print("Feature names:", california.feature_names)
print("First 5 rows of data:\n", california.data[:5])
print("Target shape:", california.target.shape)
print("First 5 target values:", california.target[:5])

X,y = california.data, california.target

# 데이터 분할
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# 피드백1 : random state = 42에서 42가 의미하는 바는?
"""
random_state 매개변수는 데이터 분할 시 난수 생성기의 시드를 설정하는 데 사용됩니다. 
이 값은 재현 가능성을 보장하기 위해 중요합니다. 즉, 동일한 random_state 값을 사용하면 동일한 입력 데이터에 대해 항상 동일한 결과를 얻을 수 있습니다. 
random_state=42는 특정한 의미를 가지는 것은 아니지만, 예제 코드나 문서에서 자주 사용되는 숫자 중 하나입니다.
"""
print(f"len(X_train):{len(X_train)})")

# 선형 모델 학습
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# 모델 예측
y_pred = lin_reg.predict(X_test)
print(f"y_pred:{y_pred}")

# 예측 결과 평가 / MSE & R^2 계산
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"mse:{mse}")
print(f"R^2 Score:{r2}")

# 시각화
plt.scatter(y_test,y_pred)
plt.plot(X_test,y_pred, color="red", linewidth=2, label="Prediction")  #피드백 2: 그래프 해석? 왜 label이 여러개로 나오는건지?
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()

"""
과제 2: 분류 분석
목표: 특정 데이터에 대한 나이브 베이즈 분류 모델을 구축하고 평가합니다.
데이터셋: Wine dataset을 사용합니다.
내용:
1. 데이터를 불러오고, 독립 변수(X)와 종속 변수(y)를 설정합니다.
2. 훈련 데이터와 테스트 데이터로 분할합니다.
3. 나이브 베이즈 모델을 학습시키고, 테스트 데이터에 대해 예측합니다.
4. 예측 결과를 평가하고, 정확도와 혼동 행렬을 계산합니다.
5. 혼동 행렬을 시각화합니다
"""

from sklearn.datasets import load_wine
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

# 데이터 분할
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# 나이브 베이즈 모델 모델 학습
nb = GaussianNB()
nb.fit(X_wine_train, y_wine_train)

# 모델 예측
y_wine_pred = nb.predict(X_wine_test)

# 평가
accuracy_wine = accuracy_score(y_wine_test,y_wine_pred)
confusion_matrix = confusion_matrix(y_wine_test,y_wine_pred)

print(f"Accuracy: {accuracy_wine}")
print(f"Confusion matrix:\n{confusion_matrix}")

# 혼동 행렬 시각화
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

"""
과제 3: 교차 검증

목표: 데이터에 대해 교차 검증을 수행하여 모델 성능을 평가합니다.
데이터셋: Diabetes dataset을 사용합니다.
내용:
데이터를 불러오고, 독립 변수(X)와 종속 변수(y)를 설정합니다.
나이브 베이즈 모델을 사용하여 5-겹 교차 검증을 수행합니다.
교차 검증 점수를 출력하고 평균 점수를 계산합니다.
교차 검증 결과를 시각화합니다.
"""

from sklearn.datasets import load_diabetes

# 데이터 로드
diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

X_dia_train, X_dia_test, y_dia_train, y_dia_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
scores_dia = cross_val_score(nb, X_dia_train, y_dia_train, cv=5)

print(f"Cross-validation scores : {scores_dia}")
print(f"Mean CV score: {scores_dia.mean()}")

# 시각화
plt.plot(range(1,len(scores_dia)+1), scores_dia, marker = "o", linestyle="--", color="b")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-validation scores")
plt.show()

# 피드백 UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
