# License Plate Recognition
<img src="readmeimg.png"/>
Libraries dependancies:
  <ul>
  <li>Tensorflow</li>
  <li>Numpy</li>
  <li>cv2</li>
  <li>imutils</li>
  </ul>
  
  <strong>You can run the demo by running "python3 finalPrototype.py"</strong>
  
  <p>In Yolo training folder, there are some cfg file, weights, python code we used to train our 2 yolos</p>
  <p>In CNN training folder, there is the python code we used to train our CNN for character recognition</p>
  <p>You can donwload pb files, yolo.weights and datasets here : https://drive.google.com/drive/folders/17gxw7tv7jy3KgJFhQiHX0IilYObFbIJp?usp=sharing </p>
 <p> More informations : https://medium.com/@theophilebuyssens/license-plate-recognition-using-opencv-yolo-and-keras-f5bfe03afc65 </p>    
 ### Criação de ambiente conda 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#criar ambiente python 3.7.10  

`conda create --prefix /media/jones/datarec/lpr/env python=3.7.10`

#ativar ambiente 
`conda activate /media/jones/datarec/lpr/env`
#instalar tensorflow 

pip install tensorflow-gpu==1.14.0

#instalar keras 

pip install keras==2.2.4 

pip install opencv-python==3.4.2.17  
pip install opencv-contrib-python==3.4.2.17  
pip install Cython --install-option="--no-cython-compile"   

#ir para a pasta do darkflow pra instala-lo:
## https://github.com/thtrieu/darkflow  
## procedimentos pra correcao de problema na instalacao local do darkflow:
####https://github.com/TheophileBuy/LicensePlateRecognition/issues/2
####Just build the Cython extensions in place. NOTE: If installing this way you will have to use ./flow in the cloned darkflow directory instead of flow as darkflow is not installed globally.
`python3 setup.py build_ext --inplace`  
####Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
`pip install -e .`   
`Install with pip globally`  
`pip install .`  

###cd /media/jones/datarec/lpr/fontes/ocr/darkflow/darkflow APENAS COMENTADO, NAO UTILIZAR    
pip install .  
pip install imutils  
pip install -Iv h5py==2.10.0  
pip install --upgrade Pillow  



### lembretes:
problema ao definir o LR. encontrada issue no proprio repo:  
Darkflow does not use the learning rate in .cfg. Use --lr instead.  
`https://github.com/thtrieu/darkflow/issues/515#issuecomment-356474112`

### comear a treinar a partir de um checkpoint
* Transformar os arquivos de ckp para o formator protobuf (.pb)

`https://github.com/thtrieu/darkflow/issues/869#issuecomment-412757194`
```
python flow.py
--model
cfg/yolo-character.cfg
--load
-1
--savepb
```
* Definir o parâmetro load no train-character.py (utiliza o ultimo checkpoint):  
`"load": -1`

## DS-CHAR-1
* DS-CHAR-1_train: 1.359 plates  


## DS-CHAR-2
* DS-CHAR-2_train: 4.731 plates
* DS-CHAR-2_test: 1.242 plates / 8.694 chars  

## DS-CHAR-RECOG-1
* DS-CHAR-RECOG-1_train: 22958 chars
* DS-CHAR-RECOG-1_test: 5740  chars

## DS-CHAR-RECOG-2
* DS-CHAR-RECOG-1_train: 22958 chars
* DS-CHAR-RECOG-1_test: 5740  chars
* Letra I agrupada com o numero 1 e letra O agrupada com numero 0




# Modelos
### Yolov2-ceia-v3  
* model size: width=608 height=608
* learning_rate=0.001
* burn_in=2000
* max_batches = 500200
* policy=steps
* steps=10000,20000,30000
* scales=.1,.1,.1
* batch = 8
* epoch = 200
* moving ave loss ~= 6.4
* train DS-CHAR-1_train  


### Yolov2-ceia-v4
* model size: width=416 height=416
* learning_rate=0.001
* burn_in=2000
* max_batches = 500200
* policy=steps
* steps=10000,20000,30000
* scales=.1,.1,.1
* batch = 8
* epoch = 200
* moving ave loss = 3.683196278051683
* train DS-CHAR-1_train  

### Yolov2-ceia-v6
* model size: width=416 height=416
* learning_rate=0.001
* burn_in=5000
* max_batches = 500200
* policy=steps
* steps=10000,20000
* scales=.1,.1
* batch = 8
* epoch = 100
* moving ave loss = 6.2
* train DS-CHAR-2_train

### Yolov2-ceia-v7
* model size: width=320 height=320
* learning_rate=0.001
* burn_in=2000
* max_batches = 500200
* policy=steps
* steps=20000,50000, 100000
* scales=.1,.1,.9
* batch = 16
* epoch = 120
* moving ave loss = 4
* train DS-CHAR-2_train


# Resultados Segmentação
* Total de amostras: 8694 chars  
* IOU = 0.5
* DS-CHAR-2_test  


Modelo | Precision | Recall | True Positive | False Positive | False Negative  
------------ | --------- | ------------- | --------- | ------------- | -------------  
Baseline Yolov2 | 0.90 | 0.41 | 3522 | 269 | 5172  
Yolov2-ceia-v3 | 0.98 | 0.99 | 8597 | 149 | 97  
Yolov2-ceia-v4 | 0.99 | 0.99 | 8596 | 130 | 98
Yolov2-ceia-v6 | **0.99** | **0.99** | **8634** | **103** | **60**
Yolov2-ceia-v7 | 0.98 | **0.99** | 8624 | 147 | 70


# Resultados Char Recognition
* DS-CHAR-RECOG-1_train

### char_recog_ceia_1
* model size: width=42 height=63
* epoch = 180
* loss = 0.05
* train DS-CHAR-RECOG-1_train

### char_recog_ceia_2
* model size: width=42 height=63
* epoch = 180
* loss = 0.05
* 2 layers dropout 0.5
* train DS-CHAR-RECOG-1_train


## Ceia_ResNet20v1_model.101
* model size: width=42 height=63
* epoch = 146
* best epoch = 102
* train DS-CHAR-RECOG-1_train
* ResNet20 v1


### char_recog_ceia_3
* model size: width=42 height=63
* epoch = 300
* loss = 0.05
* 2 layers dropout 0.5
* train DS-CHAR-RECOG-2_train


### ceia_char_recog_2_ResNet20v1_model.086
* model size: width=42 height=63
* epoch = 200
* best epoch = 102
* train DS-CHAR-RECOG-2_train

# Resultados Segmentação
* Total de amostras: 8694 chars  
* IOU = 0.5
* DS-CHAR-2_test 



Modelo | loss | acc   
------------ | --------- | -------------   
char_recog_ceia_1 | 2.67 | 0.8517  
char_recog_ceia_2 | 0.9905 | 0.9024 
Ceia_ResNet20v1_model | 0.5065| 0.93399  
char_recog_ceia_3 | 0.85 | 0.9123
ceia_char_recog_2_ResNet20v1_model.086 | 0.4451| **0.94283**  
ceia_char_recog_3_ResNet29v2_model | 0.4534 | 0.9412  