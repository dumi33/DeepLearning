# 가중치 규제 추가 

# 간단한 모델이 복잡한 모델보다 덜 과대적합될 가능성이 높다. 

from keras import regularizers

model = models.Sequential() # 여러 층을 두겠다. 
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), # L2규제를 두겠다. # 가중치 행렬의 모든 원소를 제곱하고 0.001을 곱하여 전체손실에 더한다. 
                         activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                      activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# L2 규제를 사용한 모델이 사용하지 않은 모델과 비교하여 훨씬 과대적합에 잘 견딘다. 
