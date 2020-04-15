import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D( 64, (3,3), activation='relu'))
model.add(Conv2D( 64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE)

score_train =model.evaluate(x_train,y_train)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0],score_train[1]*100))
score_test =model.evaluate(x_test,y_test)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0],score_test[1]*100))