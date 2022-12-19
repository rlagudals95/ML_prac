import tensorflow as tf
import matplotlib.pyplot as plt

# 구글에서 제공하는 mock 데이터
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

# 0 에 가까울 수록 흰색 255에 가까울 수록 검은색

# print(trainX.shape) (60000, 28, 28) 28개의 길이의 튜플(배열)이 28개로 이뤄진 이미지가 6만개
# trainY = 정답

# print(trainY) [9 0 0 ... 3 0 5] 정답이들어있는 리스트 label

class_names = ['T-shirt/top', 'Trouser ', 'Pullover', 'Dress ',
               'Coat', 'Sandal', 'Shirt', 'Sneaker ', 'Bag', 'Ankle boot']

# image를 눈으로 확인하는 방법
# plt.imshow(trainX[1])
# plt.show()

# 모델 만들기
# relu 음수를 지워줌

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(), # 2~d 데이터를 1열로 > 2행의 배열을 1행으로 output레이어 디자인
    # 이미지가 살짝만 달라져도 Flatten 값이 달라져서 응용력 없는 이미지 레이어가 됨
    # 해결방법 convolutional layer > 이미지에서 중요한 정보를 추려서(kernel을 이용) 복사본 20장을 만든다.
    # kernel을 어떻게 디자인 하느냐에 따라 이미지가 달라짐
    # translation invariance - 이미지의 위치가 달라지면 학습에 문제가 될 수 있음 > 이미지 pooling layer도 같이 이용 > 중요한 부분을 가운데로 움직여서 이미지 축소
    # 이미지의 중요한 feature 특성이 담겨 있고 그것으로 학습

    tf.keras.layers.Dense(10, activation="softmax"),# 마지막 노드는 activation은 없어도 되는데 sofgmax는 0 ~ 1의 확률 예측
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


# sparse_categorical_crossentropy 여러가지 카테고리 예측문제에서 쓰는 loss 함수

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# loss는 낮을수록 accuracy가 높을 수록 좋음
model.fit(trainX, trainY, epochs=5)