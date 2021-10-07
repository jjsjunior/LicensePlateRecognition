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

## DS-CHAR-1
* 1.359 plates

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


# Resultados
* Total de amostras: 938 chars  
* IOU = 0.5
* DS-CHAR-1  


Modelo | Precision | Recall | True Positive | False Positive | False Negative  
------------ | --------- | ------------- | --------- | ------------- | -------------  
Baseline Yolov2 | 0.90 | 0.77 | 725 | 81 | 213  
Yolov2-ceia-v3 | 0.99 | 1.0 | 936 | 8 | 2  
Yolov2-ceia-v4 | **1.0** | **1.0** | **937** | **2** | **1**