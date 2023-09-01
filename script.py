#imports
import numpy as np
from   numpy import expand_dims

import pandas as pd
import cv2    as cv
import random
import os
import glob
import csv
import statistics
import logging

from io import BytesIO

from PIL import Image

from itertools import islice

import tensorflow                as tf
from   tensorflow                import keras, numpy_function

from keras                     import layers
from keras.callbacks           import EarlyStopping,ModelCheckpoint
from keras.applications.vgg16  import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator

from sklearn                 import metrics, svm
from sklearn.utils           import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import ConfusionMatrixDisplay
from sklearn.metrics         import confusion_matrix

#augmentation
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report
# from yellowbrick.classifier import ConfusionMatrix

######################################################################################

PATH             = './dataset/'
norm             = "N"
doente           = "AAE"
left             = "Torax_LAT_E"
right            = "Torax_LAT_D"
stop_n = stop_s  = 30
current_s        = 6
# shape_w, shape_h = 2928, 2328
shape_w, shape_h = 224,224
number_of_folds  = 4 #4-fold cross validation + holdout set
np.random.seed(42)


###Log

def log(path, file):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        print("sla??")
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format, force=True)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


#######################################################################################
## Manuseio de Imagens

def openImgs(classe,lateralidade):
  r_v = []
  for subdir, dirs, files in os.walk(PATH+classe):
    if (files and lateralidade in subdir):
      r_v += [subdir+'/'+files[0]]
  return shuffle(r_v, random_state = 42)

def openImgsShuffled(classe,lateralidade):
  r_v = []
  for subdir, dirs, files in os.walk(PATH+classe):
    if (files and lateralidade in subdir):
      r_v += [subdir+'/'+files[0]]
  return shuffle(r_v, random_state = 42)

def rotation(img):
  dim = img.shape[:2]
  center = (dim[0]/2, dim[1]/2)
  rot = cv.getRotationMatrix2D(center, 15, 1)
  rotated = cv.warpAffine(img, rot,dim)
  # cv2_imshow(rotated)
  return rotated

def translation(img):
  T = np.float32([[1, 0, 0], [0, 1, -50]])
  translated = cv.warpAffine(img, T, img.shape[:2])
  # cv2_imshow(translated)
  return translated

def enbrighten(img):
  sam = expand_dims(img, 0)
  imageDataGenerator_obj = ImageDataGenerator(brightness_range=[0.9,0.9])
  iterator = imageDataGenerator_obj.flow(sam, batch_size=1)
  result = iterator.next()[0].astype('uint8')
  # cv2_imshow(result)
  return result

def change_contrast(img):
  contrast = tf.keras.layers.RandomContrast(0.3, seed = 42)
  # cv2_imshow(contrast(img).numpy().astype('uint8'))
  imageDataGenerator_obj = ImageDataGenerator(brightness_range=[0.2,1.0])
  # cv2_imshow(contrast(img).numpy().astype('uint8'))
  return contrast(img).numpy().astype('uint8')

def join(stop_n, stop_s, lateralidade=left):
  X=[] # Images
  Y=[] # type of image (normal or AE)

  for f in islice(openImgs(norm, lateralidade), 0, stop_n):
    X.append(cv.resize(cv.imread(f), (shape_w, shape_h))) #Original image 'normal'
    Y += [0]


  for f in islice(openImgs(doente, lateralidade), 0, stop_s):
    X.append(cv.resize(cv.imread(f), (shape_w, shape_h))) #Original image 'atrial enlargement'
    Y += [1]

  return np.array(X), np.array(Y)


def shuffle_Xy(stop_n, stop_s):
  X, y = join(stop_n, stop_s)

  # Shuffle two lists with same order
  # Using zip() + * operator + shuffle()
  temp = list(zip(X, y))
  random.shuffle(temp)
  auxX, auxY = zip(*temp)
  # res1 and res2 come out as tuples, and so must be converted to lists.
  X, y = list(auxX), list(auxY)
  return X, y



#######################################################################################
## Definicao de redes

def custom_InceptionV3():
  iv3 = keras.applications.InceptionV3(input_shape=(shape_w, shape_h, 3), include_top=False, weights="imagenet");

  last_layer  = iv3.get_layer('mixed10')
  last_output = last_layer.output
  extraLayer = tf.keras.layers.GlobalMaxPooling2D()(last_output)
  extraLayer = tf.keras.layers.Dense(512, activation='relu')(extraLayer)
  extraLayer = tf.keras.layers.Dropout(0.5)(extraLayer)
  extraLayer = tf.keras.layers.Dense(2, activation='sigmoid')(extraLayer)

  model = keras.Model(iv3.input, extraLayer)

  return model

    # model = custom_InceptionV3()
    # model.summary()


def custom_ResNet50V2():
  resnet = keras.applications.ResNet50V2(input_shape=(shape_w, shape_h, 3), include_top=False, weights='imagenet');

  last_layer = resnet.get_layer('conv5_block3_out')
  last_output = last_layer.output
  extraLayer = tf.keras.layers.GlobalMaxPooling2D()(last_output)
  extraLayer = tf.keras.layers.Dense(512, activation='relu')(extraLayer)
  extraLayer = tf.keras.layers.Dropout(0.5)(extraLayer)
  extraLayer = tf.keras.layers.Dense(2, activation='sigmoid')(extraLayer)

  resnet = keras.Model(resnet.input, extraLayer)
  return resnet

    # model = custom_ResNet50V2()
    # model.summary()


#######################################################################################
## Validacao
def custom_kfold(X, y, model, name, lr, numfolds, loss, epochs):
  KF = KFold(n_splits=numfolds, shuffle=True)
  vecY_predTreino = []
  vecY_predVal = []
  vecAccTreino = []
  vecAccVal = []
  vecY_val = []
  vecY_test = []
  lossValVec = []
  lossTestVec = []
  confMatrix = []

  checkpoint_path = './checkpoints/'
  # checkpoint_name = name+'_pesos.tf'
  checkpoint_name = name + "pesos.hdf5"
  checkpoint_full_path = checkpoint_path + checkpoint_name

  optimizer = keras.optimizers.Adam(learning_rate=lr)
  

  callback_list = []

  for index, (train_index, test_index) in enumerate(KF.split(X)):
    # print(f"index: {index}, t_i: {train_index}, test_index: {test_index}")
    # X_test = X[] ->>> SEM TESTE EM PRINCIPIO
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[test_index], y[test_index]
    #sem teste em principio

    ############## infos callbacks
    checkpoint = ModelCheckpoint(filepath=checkpoint_full_path, monitor='accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq=1)
    early = EarlyStopping(monitor='accuracy', min_delta=0, patience=5, verbose=0, mode='auto')
    #patience era 20
    callback_list = [checkpoint, early]
    ############################################################################
    model.compile(loss=loss, optimizer=optimizer\
      , metrics=['accuracy'])
    
    #UMA EPOCA PRA TESTAR!!!
    # history = model.fit(X_train, y_train, epochs=1, validation_split = 0, verbose=0)
    history = model.fit(X_train, y_train, epochs=epochs, validation_split = 0\
      , verbose=0, callbacks=callback_list)
    # print(history.history)
    lossVal, accVal = model.evaluate(X_val, y_val, callbacks=callback_list)
    print("========")#VER O ACCVAL
    # print(lossVal, accVal)
    y_predV = model.predict(X_val)
    y_predV = np.argmax(y_predV, axis=1)
    vecAccVal.append(accVal)
    vecY_predVal.append(y_predV)
    vecY_val.append(y_val)

    lossValVec.append(lossVal)

    model.load_weights(checkpoint_full_path)


    # lossTest, accTest = model.evaluate(X_test, y_test)
  #   y_predT = model.predict(X_test)
  #   y_predT = np.argmax(y_predT, axis=1)
  #   vecAccT.append(accTest)
  #   vecY_predT.append(y_predT)
  #   vecY_test.append(y_test)

  #   lossTestVec.append(lossTest)
    tf.keras.backend.clear_session()

    print(y_predV)
  return vecY_predVal, vecAccVal, vecY_val, lossValVec


def main(network, networkName):
  resAux = [[],[],[],[],[],[]]
  # vecAccV, vecY_predV, vecY_val, vecAccT, vecY_predT, vecY_test = [], [], [], [], [], []
  vecAccV, vecY_predV, vecY_val, vecAccT, vecY_predT, vecY_test, lossValVec, lossTestVec = runLOO(network, networkName, learning_rate, resAux)
  # resultados = pd.DataFrame({'Accuracy Test': vecAT, 'Sensibility': resAux[0], 'Specificity': resAux[1],
  #                             'Precision': resAux[2], 'Recall': resAux[3], 'F1Score': resAux[4]})
  # print(len(vecAccV), len(vecY_predV), len(vecY_val), len(vecAccT), len(vecY_predT), len(vecY_test))
  # resultados = pd.DataFrame({'vecAccV': vecAccV, 'vecY_predV': vecY_predV.flatten(), 'vecY_val': vecY_val.flatten(),
  #                             'vecAccT': vecAccT, 'vecY_predT': vecY_predT, 'vecY_test': vecY_test})

  return vecAccV, vecY_predV, vecY_val, vecAccT, vecY_predT, vecY_test, lossValVec, lossTestVec


#######################################################################################
## Metricas e Logs
#plot
def displayHistory(history):
  figure, axis = plt.subplots(nrows=1, ncols=2)
  axis[0].plot(history.history['accuracy'])
  # axis[0].plot(history.history['val_accuracy'])
  axis[0].set_title('model accuracy')
  #plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  axis[1].plot(history.history['loss'])
  # axis[1].plot(history.history['val_loss'])
  axis[1].set_title('model loss')
  #plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  plt.show()

def confusionMatrix(y_test, y_pred):

  cm   = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix = cm)

  disp.plot(cmap = plt.cm.Blues)
  plt.show()

def calcConf (y_test, y_pred, confusionM):
  if ((y_test == y_pred) and (y_test == 1)):
    aux = confusionM[0]
    confusionM[0] = aux + 1
  if ((y_test == y_pred) and (y_test == 0)):
    aux = confusionM[1]
    confusionM[1] = aux + 1
  if ((y_test != y_pred) and (y_test == 1)):
    aux = confusionM[2]
    confusionM[2] = aux + 1
  if ((y_test != y_pred) and (y_test == 0)):
    aux = confusionM[3]
    confusionM[3] = aux + 1

#confusionM[0] -> True Positive
#confusionM[1] -> True Negative
#confusionM[2] -> False Positive
#confusionM[3] -> False Negative

def sensitivity(truePositive, trueNegative, falsePositive, falseNegative):
  if (truePositive + falseNegative) == 0:
    return 0
  return (truePositive / (truePositive + falseNegative))

def specificity(truePositive, trueNegative, falsePositive, falseNegative):
  if (trueNegative + falsePositive) == 0:
    return 0
  return (trueNegative / (trueNegative + falsePositive))

def precision(y_test, y_pred):
  m = keras.metrics.Precision()
  m.update_state(y_test, y_pred)
  return (m.result().numpy())

def recall(y_test, y_pred):
  m = keras.metrics.Recall()
  m.update_state(y_test, y_pred)
  return (m.result().numpy())

def f1Score(y_test, y_pred):
  p = precision(y_test, y_pred)
  r = recall(y_test, y_pred)

  if ((p or r) == 0):
    return 0
  return ( 2 * ((p*r) / (p+r)) )


def metrics (metricVec, history, y_test, y_pred, confusionM):
  calcConf(y_test, y_pred, confusionM)
  # displayHistory(history)
  # confusionMatrix(y_test, y_pred)
  #metricVec.append(["CnfM", confusionMatrix(y_test, y_pred)])[0]
  metricVec[0].append([sensitivity(confusionM[0], confusionM[1], confusionM[2], confusionM[3])])
  metricVec[1].append([specificity(confusionM[0], confusionM[1], confusionM[2], confusionM[3])])
  if not((confusionM[0] == 0)):
    metricVec[2].append([precision(y_test, y_pred)])
    metricVec[3].append([recall(y_test, y_pred)])
    metricVec[4].append([f1Score(y_test, y_pred)])

#removendo o logger
def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)



##################### Execucao propriamente dita

#-> inception v3
learning_rateIncep   = 0.001
optIncep             = keras.optimizers.Adam(learning_rate=learning_rateIncep)
EPOCHSIncep          = 20
number_of_foldsIncep = 60
LOSSIncep            = tf.keras.losses.sparse_categorical_crossentropy
num_repIncep         = 1


def validacoesInception():
    vecAccV, vecY_predV, vecY_val, vecAccT, vecY_predT, vecY_test, lossValVec, lossTestVec = main(custom_InceptionV3(), "InceptionV3")
    with open('drive/MyDrive/TCC/metricas/inception00104.txt', 'w') as writefile:
        writefile.write(str(statistics.mean(vecAccV))+"\n")
        writefile.write(str(precision(vecY_predV, vecY_val))+"\n")
        writefile.write(str(recall(vecY_predV, vecY_val))+"\n")
        writefile.write(str(f1Score(vecY_predV, vecY_val))+"\n")
        writefile.write(str(statistics.mean(vecAccT))+"\n")
        writefile.write(str(precision(vecY_predT, vecY_test))+"\n")
        writefile.write(str(recall(vecY_predT, vecY_test))+"\n")
        writefile.write(str(f1Score(vecY_predT, vecY_test))+"\n")
    with open('drive/MyDrive/TCC/metricas/inception001loss02.txt', 'w') as writefile:
        writefile.write(str(lossValVec)+"\n")
        writefile.write(str(lossTestVec)+"\n")

    #plot de treino
    cm = confusion_matrix(np.array(vecY_val).flatten(), np.array(vecY_predV).flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.show()
    #plot de teste
    cm = confusion_matrix(vecY_test, vecY_predT)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.show()




#-> ResNet
learning_rateRes   = 0.001
optRes             = keras.optimizers.Adam(learning_rate=learning_rateRes)
EPOCHSRes          = 20
number_of_foldsRes = 60
LOSSRes            = tf.keras.losses.sparse_categorical_crossentropy
num_repRes         = 1
# vecAccV, vecY_predV, vecY_val, vecAccT, vecY_predT, vecY_test, lossValVec, lossTestVec = main(custom_ResNet50V2(), "ResNet50V2")
def resultados(vecAccV, vecY_predV, vecY_val, lossValVec, name):
    # , vecAccT, vecY_predT, vecY_test, lossTestVec):
    print(vecAccV)
    with open('./results/'+name+".txt", 'w+') as writefile:
        writefile.write("Mean accuracy: " + str(statistics.mean(vecAccV))+"\n")
        writefile.write("Precision: " + str(precision(vecY_predV, vecY_val))+"\n")
        writefile.write("Recall: " + str(recall(vecY_predV, vecY_val))+"\n")
        writefile.write("F1Score: " + str(f1Score(vecY_predV, vecY_val))+"\n")
        # writefile.write(str(statistics.mean(vecAccT))+"\n")
        # writefile.write(str(precision(vecY_predT, vecY_test))+"\n")
        # writefile.write(str(recall(vecY_predT, vecY_test))+"\n")
        # writefile.write(str(f1Score(vecY_predT, vecY_test))+"\n")
    with open('./results/'+name+".txt", 'a') as writefile:
        writefile.write(str(lossValVec)+"\n")
        # writefile.write(str(lossTestVec)+"\n")


    #plot de treino
    cm = confusion_matrix(np.array(vecY_val).flatten(), np.array(vecY_predV).flatten())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='g')
    # plt.show()
    plt.savefig("./results/cm"+name)
    #plot de teste
    # cm = confusion_matrix(vecY_test, vecY_predT)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues, values_format='g')
    # plt.show()






def run_inception(X,y):
  learning_rateIncep   = 0.001
  optIncep             = keras.optimizers.Adam(learning_rate=learning_rateIncep)
  EPOCHSIncep          = 1
  # number_of_foldsIncep = 60
  number_of_foldsIncep = 4
  LOSSIncep            = tf.keras.losses.sparse_categorical_crossentropy
  num_repIncep         = 1
  # vecAccV, vecY_predV, vecY_val, vecAccT, \
  #   vecY_predT, vecY_test, lossValVec, lossTestVec = \
  #     main(custom_InceptionV3(), "InceptionV3", learning_rateIncep, optIncep)
  vecAcc = []
  logg = "inceptionv3.logs"
  logger = log("./logs/", logg)

  vecY_predVal, vecAccVal, vecY_val, lossValVec = custom_kfold(X, y, custom_InceptionV3(), "inception", learning_rateIncep\
    , number_of_foldsIncep, LOSSIncep, EPOCHSIncep)
  resultados(vecAccVal, vecY_predVal, vecY_val, lossValVec, "Inception")
  # vecY_pred, vecY_test = custom_kfold(X, y, custom_InceptionV3(), "InceptionV3", learning_rateIncep\
  #     , logger, "teste", vecAcc, LOSSIncep, optIncep, EPOCHSIncep)
  # logging.shutdown()


  return

def run_resnet(X,y):
  learning_rateRes   = 0.001
  optRes             = keras.optimizers.Adam(learning_rate=learning_rateRes)
  EPOCHSRes          = 1
  # EPOCHSRes          = 20
  number_of_foldsRes = 60
  LOSSRes            = tf.keras.losses.sparse_categorical_crossentropy
  num_repRes         = 1
  # vecAccV, vecY_predV, vecY_val, vecAccT, \
  #   vecY_predT, vecY_test, lossValVec, lossTestVec = \
  #     main(custom_InceptionV3(), "InceptionV3", learning_rateIncep, optIncep)
  vecAcc = []
  logg = "resnet50v2.logs"
  logger = log("./logs/", logg)

  vecY_predVal, vecAccVal, vecY_val, lossValVec = custom_kfold(X, y, custom_ResNet50V2(), "resnet", learning_rateRes\
    , number_of_foldsRes, LOSSRes, EPOCHSRes)
  # resultados(vecAccVal, vecY_predVal, vecY_val, lossValVec, "ResNet")
  # vecY_pred, vecY_test = custom_kfold(X, y, custom_InceptionV3(), "InceptionV3", learning_rateIncep\
  #     , logger, "teste", vecAcc, LOSSIncep, optIncep, EPOCHSIncep)
  # logging.shutdown()


  return




if  __name__ == '__main__':
  classe = "AAE"
  print("Loading data...")

  X,y = join(30, 30)
  # for i in X:
  #   print(i)
  print(len(X))
  print("Creating networks...")
  print("k-fold...")
  print("inceptionv3")
  run_inception(X,y)
  print("ResNet")
  # run_resnet(X,y)
  

  print("results stored at file \"results.txt\"")