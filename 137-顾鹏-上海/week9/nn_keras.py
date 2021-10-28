from tensorflow.python.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import to_categorical


(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print('train_images', train_images.shape)
print('train_labels', train_labels.shape)
print('test_images', test_images.shape)
print('test_labels', test_labels.shape)

demo_image = test_images[0]
plt.imshow(demo_image, cmap = plt.cm.binary)
plt.show()

demo_model = models.Sequential()
demo_model.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
demo_model.add(layers.Dense(10, activation='softmax'))

# 模型编译方式
demo_model.compile(optimizer='rmsprop', loss="categorical_crossentropy",metrics=["accuracy"])

# 数据处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32")/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32")/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

demo_model.fit(train_images,train_labels, epochs=5,batch_size=128)

test_loss,test_acc =demo_model.evaluate(test_images,test_labels,verbose=1)
res = demo_model.predict(test_images)

for i in range(res[1].shape[0]):
    print(res[1][i])
    if (res[1][i]>0.7):
        print("the number for the picture is",i)
        break
