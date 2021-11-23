from __future__ import print_function
from tqdm import tqdm
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
import matplotlib.pyplot as plt

width = 42
height = 63
channel = 3


dict_letters_targets = {'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'j':18,'k':19,'l':20,'m':21,
                      'n':22,'p':23,'q':24,'r':25,'s':26,'t':27,'u':28,'v':29,'w':30,'x':31,'y':32,'z':33,'?':34}



def get_char_labels_idx_array():
    list_labels = []
    for idx in range(0, 34):
        list_labels.append(get_label(idx))
    return np.array(list_labels)



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


def get_label_from_array(nd_array):
    return np.array([get_label(label) for label in nd_array])



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
    erros_true_label_dict  = {}
    erros_mismatch_label_dict  = {}
    qtd_acertos = 0
    y_true_label_array = np.zeros((nrof_samples,))
    y_predited_label_array = np.zeros((nrof_samples,))
    for i in tqdm(range(qtd_steps)):
        idx_start = i*batch_size
        idx_end = idx_start + batch_size
        idx_end = nrof_samples if nrof_samples < idx_end else idx_end
        x_batch = images[idx_start:idx_end]
        y_batch = labels[idx_start:idx_end]
        x_batch = x_batch.astype('float32') / 255
        y_predicted_batch = model.predict(x_batch)
        y_predicted_classes = y_predicted_batch.argmax(axis=-1)
        y_predited_label_array[idx_start:idx_end] = y_predicted_classes
        y_true_label_array[idx_start:idx_end] = y_batch
        # y_predited_label_array[idx_start:idx_end] = get_label_from_array(y_predicted_classes)
        # y_true_label_array[idx_start:idx_end] = get_label_from_array(y_batch)
        for idx, predicted_class in enumerate(y_predicted_classes):
            if int(predicted_class)!=int(y_batch[idx]):
                y_predicted_label = get_label(int(predicted_class))
                y_true_label = get_label(int(y_batch[idx]))
                atualizar_contagem_erros(erros_true_label_dict, y_predicted_label, y_true_label)
                atualizar_contagem_erros(erros_mismatch_label_dict, y_true_label, y_predicted_label)
            else:
                qtd_acertos +=1
    possible_labels = range(0, 34)
    possible_labels = [float(possible) for possible in possible_labels]
    labels_not_predicted = [y_pred for y_pred in y_predited_label_array if y_pred not in possible_labels]
    print('labels_not_predicted: %s' % str(labels_not_predicted))
    true_labels_not_used = [y_pred for y_pred in y_true_label_array if y_pred not in possible_labels]
    print('true_labels_not_used: %s' % str(true_labels_not_used))
    name_labels_array = get_char_labels_idx_array()
    # char_confusion_matrix = confusion_matrix(y_true_label_array, y_predited_label_array,labels=name_labels_array)
    char_confusion_matrix = confusion_matrix(y_true_label_array, y_predited_label_array, labels = possible_labels)
    disp_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=char_confusion_matrix, display_labels=name_labels_array)
    print('acertos: %i' % qtd_acertos)
    print('True labels errors')
    print_errors(erros_true_label_dict)
    print('False labels errors')
    print_errors(erros_mismatch_label_dict)
    disp_conf_matrix.plot()
    plt.show()
    print('xpto')



def print_errors(errors_dict):
    for key in errors_dict.keys():
        print('true label: %s ' % key)
        list_erros_true_label = list(errors_dict[key].items())
        list_erros_true_label.sort(key=lambda x: x[1])
        for key_predited, value_predited in list_erros_true_label:
            print('predicted label: %s | qtd: %i' % (key_predited, value_predited))







def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images_test', type=str)
    parser.add_argument('--model_path', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))