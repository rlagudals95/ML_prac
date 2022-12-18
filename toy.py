import tensorflow as tf

키 = 170
신발 = 260

# 키와 신발은 연관이 있을 것이라 가정
# 신발 = 키*a + b

# a와 b엔 어떤 변수를 넣어야 할까

a = tf.Variable(0.1)
b = tf.Variable(0.2)

# 경사 하강법으로 변수를 업데이트
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 스마트한 gradient 업데이트
# 실제값과 예측값의 차이 구하기


def 손실함수():
    예측값 = 170 * a + b
    return tf.square(260 - 예측값)

for i in range(1000):
    opt.minimize(손실함수, var_list=[a, b])  # 경사하강 시작!
    print(a.numpy(), b.numpy())

# var_list = 업데이트할 변수들 목록

# ??? = a170 + b
# y = ax + b

print('예측값 : ', 170* 1.5198832  + 1.6198832)