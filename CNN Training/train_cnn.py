from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import argparse
import sys

# width = 14
# height = 21
width = 42
height = 63
channel = 3

def main(args):
    dir_images_train = args.dir_images_train
    dir_images_test = args.dir_images_test
    (train_images, train_labels), (test_images, test_labels) = load_data(dir_images_train, dir_images_test)
    train_model(train_images, train_labels, test_images, test_labels)

# dict_letters_targets = {'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'l':21,'m':22,
#                       'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,'u':30,'v':31,'w':32,'x':33,'y':34,'z':35,'?':36}


dict_letters_targets = {'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'j':18,'k':19,'l':20,'m':21,
                      'n':22,'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,'v':29,'w':30,'x':31,'y':32,'z':33,'?':34}


def load_data(dir_images_train, dir_images_test):
        images = np.array([]).reshape(0,height,width, channel)
        labels = np.array([])
        
        ################ Data in  ./AUG then in a folder with label name, example : ./AUG/A for A images #############
        directories = [x[0] for x in os.walk(dir_images_train)][2:]
        print(directories)
        for directory in directories:
                filelist = glob.glob(directory+'/*.jpg')
                sub_images = np.array([np.array(Image.open(fname).resize((width, height), Image.NEAREST)) for fname in filelist])
                # sub_labels = [int(directory[-2:])]*len(sub_images)
                letter_label_str = directory[-1:].lower()
                if letter_label_str=='?':
                    continue
                if letter_label_str.isdigit():
                    letter_label = int(letter_label_str)
                else:
                    # letter_label = ord(letter_label_str)-96+9
                    letter_label = dict_letters_targets[letter_label_str]
                sub_labels = [letter_label] * len(sub_images)
                images = np.append(images,sub_images, axis = 0)
                labels = np.append(labels,sub_labels, axis = 0)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
        return (X_train, y_train), (X_test, y_test)

# (train_images, train_labels), (test_images, test_labels) = load_data()
def train_model(train_images, train_labels, test_images, test_labels):
    # train_images = train_images.reshape((train_images.shape[0], height, width, channel))
    # test_images = test_images.reshape((test_images.shape[0], height, width,channel))
    train_images, test_images = train_images / 255.0, test_images / 255.0
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(37, activation='softmax'))
    model.add(layers.Dense(34, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels,batch_size=128, epochs=300)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc: %s' % str(test_acc))
    print('test_loss: %s' % str(test_loss))
    model.save("char_recog_ceia_3.h5")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images_train', type=str)
    parser.add_argument('--dir_images_test', type=str, default='0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
