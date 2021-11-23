from __future__ import print_function
from tqdm import tqdm
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import sys
import argparse

width = 42
height = 63
channel = 3


dict_letters_targets = {'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'j':18,'k':19,'l':20,'m':21,
                      'n':22,'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,'v':29,'w':30,'x':31,'y':32,'z':33,'?':34}



def load_data(dir_images_test):
    images = np.array([]).reshape(0, height, width, channel)
    labels = np.array([])
    ################ Data in  ./AUG then in a folder with label name, example : ./AUG/A for A images #############
    directories = [x[0] for x in os.walk(dir_images_test)][2:]
    print(directories)
    for directory in directories:
        filelist = glob.glob(directory + '/*.jpg')
        sub_images = np.array(
            [np.array(Image.open(fname).resize((width, height), Image.NEAREST)) for fname in filelist])
        # sub_labels = [int(directory[-2:])]*len(sub_images)
        letter_label_str = directory[-1:].lower()
        if letter_label_str == '?':
            continue
        if letter_label_str.isdigit():
            letter_label = int(letter_label_str)
        else:
            # letter_label = ord(letter_label_str)-96+9
            letter_label = dict_letters_targets[letter_label_str]
        sub_labels = [letter_label] * len(sub_images)
        images = np.append(images, sub_images, axis=0)
        labels = np.append(labels, sub_labels, axis=0)
    return images, labels



def get_label(y_predicted):
    if y_predicted < 10:
        return str(y_predicted)
    for key, value in dict_letters_targets.items():
        if value==y_predicted:
            return key


def atualizar_contagem_erros(erros_dict, classe_prevista_erro, true_class):
    if true_class in erros_dict:
        if classe_prevista_erro not in erros_dict[true_class]:
            erros_dict[true_class] = {classe_prevista_erro: 0}
        erros_dict[true_class][classe_prevista_erro] = erros_dict[true_class][classe_prevista_erro]+1
    else:
        erros_dict[true_class] = {classe_prevista_erro: 1}


def subtract_pixel_mean(images):
    x_images_mean = np.mean(images, axis=0)
    images -= x_images_mean
    return images


def main(args):
    dir_images_test = args.dir_images_test
    images, labels = load_data(dir_images_test)
    images = subtract_pixel_mean(images)
    model = keras.models.load_model(args.model_path)
    batch_size = 64
    nrof_samples = len(images)
    qtd_steps = int(nrof_samples / batch_size)+1
    erros_dict  = {}
    qtd_acertos = 0
    for i in tqdm(range(qtd_steps)):
        idx_start = i*batch_size
        idx_end = idx_start + batch_size
        idx_end = nrof_samples if nrof_samples < idx_end else idx_end
        x_batch = images[idx_start:idx_end]
        y_batch = labels[idx_start:idx_end]
        x_batch = x_batch.astype('float32') / 255
        y_predicted_batch = model.predict(x_batch)
        y_predicted_classes = y_predicted_batch.argmax(axis=-1)
        for idx,predicted_class in enumerate(y_predicted_classes):
            if int(predicted_class)!=int(y_batch[idx]):
                y_predicted_label = get_label(int(predicted_class))
                y_true_label = get_label(int(y_batch[idx]))
                atualizar_contagem_erros(erros_dict, y_predicted_label, y_true_label)
            else:
                qtd_acertos +=1
    for key in erros_dict.keys():
        print('true label: %s ' % key)
        for key_predited, value_predited in erros_dict[key].items():
            print('predicted label: %s | qtd: %i' % (key_predited, value_predited))
    print(erros_dict)
    print('acertos: %i' % qtd_acertos)







def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images_test', type=str)
    parser.add_argument('--model_path', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))