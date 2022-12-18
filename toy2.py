import tensorflow as tf

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]

# x에 2를 곱하고 1을 더 하면된다.

# model 만들기
# 옵티마이저 설정
# 손실함수를 만든다.
# 학습을 시킨다. ex) 경사하강

# 보통 RANDOM한 값을 집어 넣는다.
a = tf.Variable(0.1)
b = tf.Variable(0.1)

# 경사하강법

# 예측값과 실제값의 차이를 나타내는 수식
# mean sqaured error (예측1 - 실제1)^2 + (예측2 - 실제2)^2


def 손실함수(a, b):
    예측_y = train_x * a + b
    return tf.keras.losses.mse(train_y, 예측_y)


opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(3000):
    opt.minimize(lambda: 손실함수(a, b), var_list=[a, b])
    print(a.numpy(), b.numpy())

# 딥러닝은 Nueral Network를 만들어야한다.
# 히든 레이어가 있어야한다.