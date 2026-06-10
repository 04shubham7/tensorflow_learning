import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to hide warnings
import tensorflow as tf

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist=keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape) #(60000, 28, 28) (60000,)

#normalize the data:0,255->0,1
x_train,x_test=x_train/255.0,x_test/255.0

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(x_train[i],cmap='gray')
# plt.savefig('mnist_digits.png')

#model
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),#reduces the dimensions of the input data
    keras.layers.Dense(128,activation='relu'),#fully connected layer with 128 neurons and ReLU activation function
    keras.layers.Dense(10),#fully connected layer with 10 neurons (output layer) and no activation function (logits
])

print(model.summary())#prints the summary of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.

#alternative way to define the model
# model=keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28,28)))#we can just add a print statement to check the summary of the model after adding each layer
# print(model.summary())
# model.add(keras.layers.Dense(128,activation='relu'))
# model.add(keras.layers.Dense(10))

#loss and optimizer
#y=integer labels->sparse categorical crossentropy
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)#loss function for multi-class classification problems where the labels are provided as integers (sparse format) and the model outputs logits (raw, unnormalized scores).
optim=keras.optimizers.Adam(lr=0.001)#Adam optimizer is an adaptive learning rate optimization algorithm that combines the advantages of both the AdaGrad and RMSProp algorithms, making it well-suited for training deep neural networks.
metrics=['accuracy']#metrics to evaluate the performance of the model during training and testing. In this case, we are using accuracy as the metric to measure how well the model is performing in terms of correctly classifying the input data.

model.compile(optimizer=optim,loss=loss,metrics=metrics)

#training the model
batch_size=64
epochs=5

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2)

#evaluating the model
model.evaluate(x_test,y_test,batch_size=batch_size,verbose=2)

#prediction
probability_model=keras.Sequential([
    model,
    keras.layers.Softmax()
    ])

predictions=probability_model.predict(x_test)
pred0=predictions[0]
print(pred0)
label0=np.argmax(pred0)
print(label0)

#model+softmax
predictions=model(x_test)#model.predict(x_test,batch_size=batch_size,verbose=2)
predictions=tf.nn.softmax(predictions)
pred0=predictions[0]
print(pred0)
label0=np.argmax(pred0)
print(label0)

pred05s=predictions[5]
print(pred05s.shape)
label05s=np.argmax(pred05s,axis=1)
print(label05s)


"""
This tutorial is part of a TensorFlow series and focuses on building a first neural network using the Keras Sequential API. The video walks through the entire pipeline: preparing data, building, training, evaluating, and predicting with a model.

Key Highlights of the Process:
Data Preparation (3:02 - 5:50): The video uses the famous MNIST dataset (handwritten digits). It demonstrates loading the data, inspecting shapes, and normalizing pixel values from [0, 255] to [0, 1] for better model performance.
Building the Model (5:58 - 12:28): Using the Keras Sequential API, a model is defined with:
A Flatten layer to handle the 28x28 input.
A Dense layer with 128 units and ReLU activation.
A final Dense layer with 10 outputs (for the 10 digit classes).
Compilation & Training (12:38 - 17:20): The model is configured with the Adam optimizer and Sparse Categorical Cross-Entropy loss. 
                                        The training uses model.fit() over five epochs, achieving high accuracy (~98%).
Evaluation (17:30 - 19:10): The model.evaluate() function is used to test the model's performance on unseen test data.
Prediction (19:15 - 25:50): Several methods to make predictions are demonstrated, 
                    including using a secondary "probability model" with a Softmax layer or manually applying tf.nn.softmax to raw model outputs.
                    The video explains using np.argmax to extract the predicted digit class.

"""