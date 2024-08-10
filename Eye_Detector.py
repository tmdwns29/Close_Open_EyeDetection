import cv2
import numpy as np
from matplotlib import pyplot as plt
import keras
import os

# 이미지 그래프 그리는 함수
def plot_images(n_row:int, n_col:int, image:list[np.array]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            if n_row <= 1:
                axis = ax[j]
            else:
                axis = ax[i,j]
                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)
                axis.imshow(image[i * n_col + j])
    plt.show()
    return None

# 졸음 이미지 파일 가져오기
path1 = './ImprovementSet/Closed/'
sleep_images = list()
for image in os.listdir(path1):
    img = cv2.imread(path1+image)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR을 RGB로 변환
    sleep_images.append(img)

# 졸음 이미지 그래프 그리기
plot_images(n_row=3, n_col=5, image=sleep_images)

# 정상 이미지 불러오기
path2 = './ImprovementSet/Opened/'
normal_images = []
for image in os.listdir(path2):
    img = cv2.imread(path2 + image)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    normal_images.append(img)

# 정상 이미지 그래프 그리기
plot_images(n_row=3, n_col=5, image=normal_images)

# 학습 시킬 데이터와 정답 데이터 만들기
X = sleep_images + normal_images # 졸음이미지+정상이미지 병합
y = [[1,0]] * len(sleep_images) + [[0,1]] * len(normal_images) # 원핫 인코딩 방식처럼

# numpy로 변환
X = np.array(X)
y = np.array(y)
print(y)
print(y.shape)

# CNN모델 만들기
model = keras.models.Sequential([
    keras.Input(shape=(64, 64, 3)), # color이미지(64 x 64)

    keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=64, activation='relu', name='LAYER1'),
    keras.layers.Dense(units=32, activation='relu', name='LAYER2'),

    keras.layers.Dense(units=2, activation='softmax', name='OUTPUT')
], name='FACE_CNN')

# 모델 학습이 끝났으므로 주석 처리
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 1000번 학습된 CNN모델 생성
history = model.fit(x=X, y=y, epochs=100)
model.save('SLEEP_DETECTOR.keras')

# 성능 테스트 : 예제 파일을 읽어와서 이미지 테스트 하기
example_images = list()

# 예제 이미지 가져오기
path3 = './testSet/examples/'
for image in os.listdir(path3):
    img = cv2.imread(path3 + image)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
    example_images.append(img)

# example_images python 파일
example_images = np.array(example_images)
plot_images(n_row=2, n_col=5, image=example_images)

# 학습완료한 모델 가져오기
model_2 = keras.models.load_model('SLEEP_DETECTOR.keras')
predict_images = model_2.predict(x=example_images)
print(np.round(predict_images))

# 2 x 5 이미지 그래프 그리기
(fig, ax) = plt.subplots(2, 5, figsize=(10, 4))
for i in range(2):
    for j in range(5):
        axis = ax[i, j]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        if predict_images[i * 5 + j][0] > 0.5:
            axis.imshow(example_images[i * 5 + j])
            axis.text(0.5, 1.05, "warning!",
                      fontsize=10, color='red',
                      ha='center', va='bottom',
                      transform=axis.transAxes)
plt.show()

loss, accuracy = model_2.evaluate(X, y, verbose=0)

# Print the model's performance
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')
