import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 구글에서 제공하는 mock 데이터
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 0 에 가까울 수록 흰색 255에 가까울 수록 검은색

# print(trainX.shape) (60000, 28, 28) 28개의 길이의 튜플(배열)이 28개로 이뤄진 이미지가 6만개
# trainY = 정답

# print(trainY) [9 0 0 ... 3 0 5] 정답이들어있는 리스트 label

trainX = trainX / 255.0
textX = testX / 255.0


trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser ', 'Pullover', 'Dress ',
               'Coat', 'Sandal', 'Shirt', 'Sneaker ', 'Bag', 'Ankle boot']

# image를 눈으로 확인하는 방법
# plt.imshow(trainX[1])
# plt.show()

# 모델 만들기
# relu 음수를 지워줌

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),  # 컨볼루셔 레이어 적용 / 첫 번째 인자 - 이미지 복사본, 두 번째 커널 사이즈
    tf.keras.layers.MaxPooling2D((2, 2)),  # 이미지의 중요 정보들을 가운데로 모음
    tf.keras.layers.Flatten(),  # 2~d 데이터를 1열로 > 2행의 배열을 1행으로 output레이어 디자인
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),

    # 이미지가 살짝만 달라져도 Flatten 값이 달라져서 응용력 없는 이미지 레이어가 됨
    # 해결방법 convolutional layer > 이미지에서 중요한 정보를 추려서(kernel을 이용) 복사본 20장을 만든다.
    # kernel을 어떻게 디자인 하느냐에 따라 이미지가 달라짐
    # translation invariance - 이미지의 위치가 달라지면 학습에 문제가 될 수 있음 > 이미지 pooling layer도 같이 이용 > 중요한 부분을 가운데로 움직여서 이미지 축소
    # 이미지의 중요한 feature 특성이 담겨 있고 그것으로 학습

    # 마지막 노드는 activation은 없어도 되는데 sofgmax는 0 ~ 1의 확률 예측
    tf.keras.layers.Dense(10, activation="softmax"),
])
# sigmoid는 0인지 1인지 예측 boolean한 확률예측

# class_names 중에 맞는 확률들이 array로 나옴 [0.2, 0.4 , 0.1, .....]

# model 확인
# model.summary() / input_shape=(28,28) 첫번째 뉴럴에 인자로 줘야 출력가능

# model.summary()
# exit()

# _________________________________________________________________
#  Layer (type)                Output Shape(레이어 모양)    Param # (학습가능한 w, b갯수)
# =================================================================
#  dense (Dense)               (None, 28, 128)           3712

#  dense_1 (Dense)             (None, 28, 64)            8256

#  flatten (Flatten)           (None, 1792)              0

#  dense_2 (Dense)             (None, 10)                17930


model.summary()

# sparse_categorical_crossentropy 여러가지 카테고리 예측문제에서 쓰는 loss 함수

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=['accuracy'])

# loss는 낮을수록 accuracy가 높을 수록 좋음
# validation_data는 epochs가 한번 끝날때 마다 모델을 채점을 해줌
# epoch 마다 채점 하는 이유는 학습 오버피팅이 일어나 정해진 답만 외우는 경우를 방지!
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

# 모델을 다 만들고 학습한 상태에서 마지막으로 모델을 평가함
# 학습용 데이터 대신 처음 보는 데이터로 평가
score = model.evaluate(testX, testY)
print(score)


#모델이 트레이닝 데이터를 과도하게 학습해서 트레이닝 데이터만 잘 예측하는겁니다. 

# 그래서 새로운 데이터를 갖다줬을 때 예측을 잘할 수 없게되니 좋은 모델이라고 할 수 없겠죠. 

# overfitting이 많이 일어나는지 언제나 체크하고 이를 방지하기 위한 여러가지 방법을 도입하면 됩니다.

# 한가지는 그냥 테스트/val accuracy가 낮아지기 전에 epoch을 중지시키는 것이 가장 쉬운 방법인데
# val_accuracy를 높일 방법을 찾자
# Dense layer 추가?
# 컨볼루션 + pooling 세트 추가?
