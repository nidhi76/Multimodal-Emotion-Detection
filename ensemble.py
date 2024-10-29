import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import vlc
import time
from pathlib import Path
from random import randint
from subprocess import call
from tkinter import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import pandas as pd
import numpy as np
# text preprocessing
from nltk.tokenize import word_tokenize
import re
# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# preparing input to our model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# keras layers
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

import nltk
import pickle



#music player function
def music_player(emotion_str):
    from Music_Player import MusicPlayer
    root = Tk()
    print('\nPlaying ' + emotion_str + ' songs')
    MusicPlayer(root,emotion_str)
    root.mainloop()
    
    
def clean_text(data):
        
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    import pandas as pd

    # tokenization using nltk
    data = word_tokenize(data)
        
    return data
    
def text_analysis(text):
    # Number of labels: joy, anger, fear, sadness, neutral
    num_classes = 5

    # Number of dimensions for word embedding
    embed_num_dims = 300

    # Max input length (max number of words) 
    max_seq_len = 500

    class_names = ['angry', 'fear', 'happy', 'neutral', 'sad']

    data_train = pd.read_csv("data_train.csv", encoding='utf-8')
    data_test = pd.read_csv("data_test.csv", encoding='utf-8')

    X_train = data_train.Text
    X_test = data_test.Text

    y_train = data_train.Emotion
    y_test = data_test.Emotion

    data = data_train.append(data_test, ignore_index=True)

    texts = [' '.join(clean_text(text)) for text in data.Text]
    texts_train = [' '.join(clean_text(text)) for text in X_train]
    texts_test = [' '.join(clean_text(text)) for text in X_test]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequence_train = tokenizer.texts_to_sequences(texts_train)
    sequence_test = tokenizer.texts_to_sequences(texts_test)

    with open("myDictionary.pkl", "rb") as tf:
        index_of_words = pickle.load(tf)


    # vocab size is number of unique words + 0 index reserved for padding
    vocab_size = len(index_of_words) + 1

    #print('Number of unique words: {}'.format(len(index_of_words)))

    #X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
    #X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

    encoding = {
        'angry': 0,
        'fear': 1,
        'happy': 2,
        'neutral': 3,
        'sad': 4
    }

    # Integer labels
    #y_train = [encoding[x] for x in data_train.Emotion]

    #y_test = [encoding[x] for x in data_test.Emotion]

    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)

    #y_train

    word_embedding = np.load("embedd_matrix.npy")

    #word_embedding.shape

    # Inspect unseen words
    new_words = 0

    for word in index_of_words:
        entry = word_embedding[index_of_words[word]]
        if all(v == 0 for v in entry):
            new_words = new_words + 1

    #print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
    #print('New words found: ' + str(new_words))

    # Embedding layer before the actaul BLSTM 
    embedd_layer = Embedding(vocab_size,
                             embed_num_dims,
                             input_length = max_seq_len,
                             weights = [word_embedding],
                             trainable=False)

    # Convolution
    kernel_size = 3
    filters = 256

    model = Sequential()
    model.add(embedd_layer)
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights('text_cnn_model1.h5')

    #predictions = model.predict(X_test_pad)
    #predictions = np.argmax(predictions, axis=1)
    #predictions = [class_names[pred] for pred in predictions]

    #print("Accuracy: {:.2f}%".format(accuracy_score(data_test.Emotion, predictions) * 100))
    #print("\nF1 Score: {:.2f}".format(f1_score(data_test.Emotion, predictions, average='micro') * 100))

    message = text
    msg =[]
    msg.append(message)
    seq = tokenizer.texts_to_sequences(msg)
    padded = pad_sequences(seq, maxlen=max_seq_len)

    start_time = time.time()
    pred = model.predict(padded)
    print("pred_text : ", class_names[np.argmax(pred)])
    return pred

    #print('Message: ' + str(msg))
    #print('predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))
    
def image_analysis():
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))


    model.load_weights('Image_Model.h5')

    
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad"}


    #File to append the emotions
    with open(str(Path.cwd())+"\emotion.txt","w") as emotion_file:
                    
            # start the webcam feed
        cap = cv2.VideoCapture(0)
        now = time.time()  ###For calculate seconds of video
        future = now + 25
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            #FindingOpencv pretrained classifier loaded
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            #Converting the image to grayscaleInitially, the image is a three-layer image (i.e., RGB), So It is converted to a one-layer image (i.e., grayscale). 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Applying the face detection method on the grayscale image
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                #print(prediction)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                emotion_file.write(emotion_dict[maxindex]+"\n")
                emotion_file.flush()

            cv2.imshow('Video', cv2.resize(frame,(700,500),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
            if time.time() > future:  ##after 10 second music will play
                cv2.destroyAllWindows()
                #music_player(text)
                future = time.time() + 25
                break

                
    print("pred_face : ", emotion_dict[maxindex])
    cap.release()
    return prediction
    
def ensemble(pred_t, pred_i):
    emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad"}
    #print('Entered ensemble')
    #pred_e = pred_t + pred_i
    pred_e = (0.7384113166485311*pred_t + 0.48978950675463423*pred_i)/1.2282008234031654
    #print("pred_ensemble: ", pred_e)
    maxindex_e = int(np.argmax(pred_e))
    text = emotion_dict[maxindex_e]
    print("Ensemble :", text)
    music_player(text)
       
                
    
    
if __name__ == "__main__"  :
    print('\n Welcome to Music Player based on Multimodal Emotion Recognition \n')
    print('\n Press \'q\' to exit the music player \n')
    pred_t = text_analysis()
    pred_i = image_analysis()
    ensemble(pred_t, pred_i)
    
