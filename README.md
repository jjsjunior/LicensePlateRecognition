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
conda create --prefix /media/jones/datarec/lpr/env python=3.7.10
#ativar ambiente
conda activate /media/jones/datarec/lpr/env
#instalar tensorflow
pip install tensorflow-gpu==1.14.0
#instalar keras
pip install keras==2.2.4
pip install opencv-python==3.4.2.17
pip install opencv-contrib-python==3.4.2.17
pip install Cython --install-option="--no-cython-compile"
#vai pra pastar do darkflow pra instala-lo:
## https://github.com/thtrieu/darkflow
##cd /media/jones/datarec/lpr/fontes/ocr/darkflow/darkflow
pip install .
pip install imutils
pip install -Iv h5py==2.10.0
pip install --upgrade Pillow

