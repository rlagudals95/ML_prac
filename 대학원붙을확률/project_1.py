import tensorflow as tf
import pandas as pd  # 액셀 데이터 포맷
import numpy as np  # 파이썬 배열대신 numpy를 넣어야 학습가능

data = pd.read_csv('./gpascore.csv')

# 데이터 전처리하기, 데이터 정제화하기
# data.isnull().sum() # 빈칸에 채워주기
# data.fillna(100) # 빈칸을 임의의 값으로 채워줌
# data['gpa'] # 원하는 열만 출력
# data['gpa'].max() , min(), count() 해당 열의 최대 최소 갯수 출력

data = data.dropna()  # 빈칸일 경우 행 지워줌

y데이터 = data['admit'].values  # 결과 붙을 확률 array 형태
x데이터 = []
# iterrows = pandas로 오픈한 데이터프레임을 붙인다 > dataframe을 한 행씩 출력
for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

# 대학원에 붙을 확률 예측

# 1. 모델만들기
# keras를 쓰면 딥러닝 모델을 만들기 매우 쉬워짐
# Sequential을 쓰면 신경망 레이어들을 쉽게 만들어줌
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),  # 숫자는 히든레이어의 노드의 개수
    # 노드 갯수는 관습적으로 2의 배수 결과가 잘 나올때까지 알아서
    tf.keras.layers.Dense(64, activation='tanh'),
    # 마지막은 1개의 노드로 수렴해야한다. #확률은 0~1 사이여야한다 = sigmoid
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 2. 옵티마이저

# 함수인자 1. 옵티마이저 / 2. 손실함수(0과 1사이 확률예측은 binary_crossentropy) / 3. 딥러닝 시
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# 학습
# x는 단서, y는 답
# epochs 학습

model.fit(np.array(x데이터), np.array(y데이터), epochs=900)

# Epoch 실행횟수, loss 손실값 예측값과 실제 값의 차이 - 낮을수록좋음, 예측한 값과 실제값과 차이 - 높을수록좋음

# 예측
# 학습한 모델로 예측해보기

예측값 = model.predict([[750, 3.70, 1], [200, 1.2, 1]])

print(예측값)

# [[0.55849856]
# [0.00661278]]
