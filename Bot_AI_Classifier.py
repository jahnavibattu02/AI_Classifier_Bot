#Import required libraries
from telegram.ext import *
from io import BytesIO
import cv2 
import numpy as np
import tensorflow as tf

#Loading and Splitting the data
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.cifar10.load_data()

#Normalizing the data
X_train, X_test = X_train/255, X_test/255

#Class names in the dataset
class_names = ['Plane', 'Car', 'Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

#Convolutional Neural Network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#Funtion fot start command
def start(update, context):
    update.message.reply_text("Welcome to AI Classifier Bot..!")

#Function for Help command
def help(update, context):
    update.message.reply_text("""
    /start - Starts Conversation
    /help - Shows this message
    /train - Trains the Neural Network
    """)

#Function for train 
def train(update, context):
    update.message.reply_text('Model is being trained')
    update.message.reply_text('May takes some time...')
    model.compile(optimizer='adam',loss='sparse_categprical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save('cifar_classifier.model')
    update.message.reply_text('Training the model is Done!, You can send a photo..!')

#Function to avoid any input text 
def handle_message(update, context):
    update.message.reply_text("Please Train the model and send a picture!")

#Function to handle input photo
def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()),dtype=np.uint8)

    #Loading the Image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2,resize(img, (32,32), interpolation=cv2.INTER_AREA)

    #Prediction
    prediction = model.predict(np.array([img/255]))
    update.message.reply_test(f"In this image I can see a{class_names[np.argmax(prediction)]}")

# Main function
updater = Updater('Enter Your TOKEN Here' , use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("Start",start))
dp.add_handler(CommandHandler("Help",help))
dp.add_handler(CommandHandler("Train",train))
dp.add_handler(MessageHandler(Filters.text,handle_message))
dp.add_handler(MessageHandler(Filters.photo,handle_photo))

updater.start_polling()
updater.idle()