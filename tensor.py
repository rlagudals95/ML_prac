import tensorflow as tf

텐서 = tf.constant([3, 4, 5])
텐서2 = tf.constant([6, 7, 8])
텐서3 = tf.constant([[1, 2], [3, 4]])
텐서4 = tf.zeros([2, 2, 3])


w = tf.Variable(1.0) # variable은 항상 변경이 쉽게 된다 var
print(w.numpy())

w.assign(2) # 새 값할당

# 텐서를 배우는 이유

# x1 x2 x3라는 인풋이 있다
# w1 w1 w3
# w와 x를 곱해서 새로운 노드를 만들고 싶다..!
# python 코드를 짜기 귀찮다

# x1* w1  + x2*w2.....
# 행렬이라는 개념을 빌려와서 노드 계산을 한다!
# 텐서는 행렬과 비슷하다
