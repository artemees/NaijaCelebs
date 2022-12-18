# NaijaCelebs

![celeb](https://user-images.githubusercontent.com/59312765/208271630-7c6efa3a-de53-4e44-aaa9-871f9c313660.png)


```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

%matplotlib inline
```


```python
img = cv2.imread('./test/5-24.jpg')
img.shape
```




    (442, 437, 3)




```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape
```




    (442, 437)




```python
gray
```




    array([[255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           ...,
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)




```python
plt.imshow(gray, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x17691321000>



![output_4_1](https://user-images.githubusercontent.com/59312765/208316061-26a0f26b-3aee-4a20-b9c9-e35abaf6b8c3.png)
    



```python
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces
```




    array([[178,  27,  82,  82]])




```python
(x,y,w,h) = faces[0]
x,y,w,h
```




    (178, 27, 82, 82)




```python
face_img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
plt.imshow(face_img)
```




    <matplotlib.image.AxesImage at 0x17691913130>




    
![output_7_1](https://user-images.githubusercontent.com/59312765/208316083-b8f527f7-43c4-427d-bf76-d35f6afb3017.png)
    



```python
cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0),2)
        
        
plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()
```


    
![output_8_0](https://user-images.githubusercontent.com/59312765/208316095-164a82bf-866a-4d5b-b88a-cd1fc7c40714.png)
    



```python
%matplotlib inline
plt.imshow(roi_color, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x17690ccf0d0>




![output_9_1](https://user-images.githubusercontent.com/59312765/208316105-eebdfacc-807d-4700-9ba2-75ed37e3fba5.png)
    



```python
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                return roi_color    
```


```python
original_image = cv2.imread('./test/5-24.jpg')
plt.imshow(original_image)
```




    <matplotlib.image.AxesImage at 0x17690d59870>




![output_11_1](https://user-images.githubusercontent.com/59312765/208316122-f9215ba0-4b88-467a-8aa1-629d7db83caf.png)
    



```python
cropped_image = get_cropped_image_if_2_eyes('./test/5-24.jpg')
plt.imshow(cropped_image)
```




    <matplotlib.image.AxesImage at 0x17690dbb400>




![output_12_1](https://user-images.githubusercontent.com/59312765/208316140-bef9b8e5-40f0-424a-aded-c3ae14f65e80.png)
    



```python
path_to_data = './data/'
path_to_cr_data = './data/cropped/'
```


```python
import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
```


```python
img_dirs
```




    ['./data/Burna Boy',
     './data/Davido',
     './data/Falz',
     './data/Tems',
     './data/Tiwa Savage',
     './data/Yemi Alade']



### Create Cropped Folder


```python
import shutil
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)
```


```python
cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]   
    print(celebrity_name)
    
    celebrity_file_names_dict[celebrity_name] = []
    
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print('Creating cropped images in folder:', cropped_folder)
                
                
            cropped_file_name = celebrity_name + str(count) + '.jpg'
            cropped_file_path = cropped_folder + '/' + cropped_file_name
            
            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1                
```

    Burna Boy
    Creating cropped images in folder: ./data/cropped/Burna Boy
    Davido
    Creating cropped images in folder: ./data/cropped/Davido
    Falz
    Creating cropped images in folder: ./data/cropped/Falz
    Tems
    Creating cropped images in folder: ./data/cropped/Tems
    Tiwa Savage
    Creating cropped images in folder: ./data/cropped/Tiwa Savage
    Yemi Alade
    Creating cropped images in folder: ./data/cropped/Yemi Alade
    


```python
import numpy as np 
import matplotlib.pyplot as plt 
import pywt

def w2d(img, mode = 'haar', level=1):
    imArray = img
    
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255;
    
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;
    
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H    
```


```python
im_har = w2d(cropped_image, 'db1', 5)
plt.imshow(im_har, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x17690e4b1f0>





![output_20_1](https://user-images.githubusercontent.com/59312765/208316149-e9f8be3b-38a6-4569-bb32-d6bf67211545.png)
    



```python
celebrity_file_names_dict
```




    {'Burna Boy': ['./data/cropped/Burna Boy/Burna Boy1.jpg',
      './data/cropped/Burna Boy/Burna Boy2.jpg',
      './data/cropped/Burna Boy/Burna Boy3.jpg',
      './data/cropped/Burna Boy/Burna Boy4.jpg',
      './data/cropped/Burna Boy/Burna Boy5.jpg',
      './data/cropped/Burna Boy/Burna Boy6.jpg',
      './data/cropped/Burna Boy/Burna Boy7.jpg',
      './data/cropped/Burna Boy/Burna Boy8.jpg',
      './data/cropped/Burna Boy/Burna Boy9.jpg',
      './data/cropped/Burna Boy/Burna Boy10.jpg',
      './data/cropped/Burna Boy/Burna Boy11.jpg',
      './data/cropped/Burna Boy/Burna Boy12.jpg',
      './data/cropped/Burna Boy/Burna Boy13.jpg',
      './data/cropped/Burna Boy/Burna Boy14.jpg',
      './data/cropped/Burna Boy/Burna Boy15.jpg',
      './data/cropped/Burna Boy/Burna Boy16.jpg',
      './data/cropped/Burna Boy/Burna Boy17.jpg',
      './data/cropped/Burna Boy/Burna Boy18.jpg',
      './data/cropped/Burna Boy/Burna Boy19.jpg',
      './data/cropped/Burna Boy/Burna Boy20.jpg',
      './data/cropped/Burna Boy/Burna Boy21.jpg',
      './data/cropped/Burna Boy/Burna Boy22.jpg'],
     'Davido': ['./data/cropped/Davido/Davido1.jpg',
      './data/cropped/Davido/Davido2.jpg',
      './data/cropped/Davido/Davido3.jpg',
      './data/cropped/Davido/Davido4.jpg',
      './data/cropped/Davido/Davido5.jpg',
      './data/cropped/Davido/Davido6.jpg',
      './data/cropped/Davido/Davido7.jpg',
      './data/cropped/Davido/Davido8.jpg',
      './data/cropped/Davido/Davido9.jpg',
      './data/cropped/Davido/Davido10.jpg',
      './data/cropped/Davido/Davido11.jpg',
      './data/cropped/Davido/Davido12.jpg',
      './data/cropped/Davido/Davido13.jpg',
      './data/cropped/Davido/Davido14.jpg',
      './data/cropped/Davido/Davido15.jpg',
      './data/cropped/Davido/Davido16.jpg',
      './data/cropped/Davido/Davido17.jpg',
      './data/cropped/Davido/Davido18.jpg',
      './data/cropped/Davido/Davido19.jpg',
      './data/cropped/Davido/Davido20.jpg',
      './data/cropped/Davido/Davido21.jpg'],
     'Falz': ['./data/cropped/Falz/Falz1.jpg',
      './data/cropped/Falz/Falz2.jpg',
      './data/cropped/Falz/Falz3.jpg',
      './data/cropped/Falz/Falz4.jpg',
      './data/cropped/Falz/Falz5.jpg',
      './data/cropped/Falz/Falz6.jpg',
      './data/cropped/Falz/Falz7.jpg',
      './data/cropped/Falz/Falz8.jpg',
      './data/cropped/Falz/Falz9.jpg',
      './data/cropped/Falz/Falz10.jpg',
      './data/cropped/Falz/Falz11.jpg',
      './data/cropped/Falz/Falz12.jpg',
      './data/cropped/Falz/Falz13.jpg',
      './data/cropped/Falz/Falz14.jpg',
      './data/cropped/Falz/Falz15.jpg',
      './data/cropped/Falz/Falz16.jpg',
      './data/cropped/Falz/Falz17.jpg',
      './data/cropped/Falz/Falz18.jpg',
      './data/cropped/Falz/Falz19.jpg',
      './data/cropped/Falz/Falz20.jpg',
      './data/cropped/Falz/Falz21.jpg',
      './data/cropped/Falz/Falz22.jpg',
      './data/cropped/Falz/Falz23.jpg',
      './data/cropped/Falz/Falz24.jpg',
      './data/cropped/Falz/Falz25.jpg',
      './data/cropped/Falz/Falz26.jpg',
      './data/cropped/Falz/Falz27.jpg',
      './data/cropped/Falz/Falz28.jpg',
      './data/cropped/Falz/Falz29.jpg'],
     'Tems': ['./data/cropped/Tems/Tems1.jpg',
      './data/cropped/Tems/Tems2.jpg',
      './data/cropped/Tems/Tems3.jpg',
      './data/cropped/Tems/Tems4.jpg',
      './data/cropped/Tems/Tems5.jpg',
      './data/cropped/Tems/Tems6.jpg',
      './data/cropped/Tems/Tems7.jpg',
      './data/cropped/Tems/Tems8.jpg',
      './data/cropped/Tems/Tems9.jpg',
      './data/cropped/Tems/Tems10.jpg',
      './data/cropped/Tems/Tems11.jpg',
      './data/cropped/Tems/Tems12.jpg',
      './data/cropped/Tems/Tems13.jpg',
      './data/cropped/Tems/Tems14.jpg',
      './data/cropped/Tems/Tems15.jpg',
      './data/cropped/Tems/Tems16.jpg',
      './data/cropped/Tems/Tems17.jpg',
      './data/cropped/Tems/Tems18.jpg',
      './data/cropped/Tems/Tems19.jpg',
      './data/cropped/Tems/Tems20.jpg',
      './data/cropped/Tems/Tems21.jpg',
      './data/cropped/Tems/Tems22.jpg',
      './data/cropped/Tems/Tems23.jpg',
      './data/cropped/Tems/Tems24.jpg'],
     'Tiwa Savage': ['./data/cropped/Tiwa Savage/Tiwa Savage1.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage2.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage3.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage4.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage5.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage6.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage7.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage8.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage9.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage10.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage11.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage12.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage13.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage14.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage15.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage16.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage17.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage18.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage19.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage20.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage21.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage22.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage23.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage24.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage25.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage26.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage27.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage28.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage29.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage30.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage31.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage32.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage33.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage34.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage35.jpg',
      './data/cropped/Tiwa Savage/Tiwa Savage36.jpg'],
     'Yemi Alade': ['./data/cropped/Yemi Alade/Yemi Alade1.jpg',
      './data/cropped/Yemi Alade/Yemi Alade2.jpg',
      './data/cropped/Yemi Alade/Yemi Alade3.jpg',
      './data/cropped/Yemi Alade/Yemi Alade4.jpg',
      './data/cropped/Yemi Alade/Yemi Alade5.jpg',
      './data/cropped/Yemi Alade/Yemi Alade6.jpg',
      './data/cropped/Yemi Alade/Yemi Alade7.jpg',
      './data/cropped/Yemi Alade/Yemi Alade8.jpg',
      './data/cropped/Yemi Alade/Yemi Alade9.jpg',
      './data/cropped/Yemi Alade/Yemi Alade10.jpg',
      './data/cropped/Yemi Alade/Yemi Alade11.jpg',
      './data/cropped/Yemi Alade/Yemi Alade12.jpg',
      './data/cropped/Yemi Alade/Yemi Alade13.jpg',
      './data/cropped/Yemi Alade/Yemi Alade14.jpg',
      './data/cropped/Yemi Alade/Yemi Alade15.jpg',
      './data/cropped/Yemi Alade/Yemi Alade16.jpg',
      './data/cropped/Yemi Alade/Yemi Alade17.jpg',
      './data/cropped/Yemi Alade/Yemi Alade18.jpg',
      './data/cropped/Yemi Alade/Yemi Alade19.jpg',
      './data/cropped/Yemi Alade/Yemi Alade20.jpg',
      './data/cropped/Yemi Alade/Yemi Alade21.jpg',
      './data/cropped/Yemi Alade/Yemi Alade22.jpg',
      './data/cropped/Yemi Alade/Yemi Alade23.jpg',
      './data/cropped/Yemi Alade/Yemi Alade24.jpg',
      './data/cropped/Yemi Alade/Yemi Alade25.jpg',
      './data/cropped/Yemi Alade/Yemi Alade26.jpg',
      './data/cropped/Yemi Alade/Yemi Alade27.jpg',
      './data/cropped/Yemi Alade/Yemi Alade28.jpg',
      './data/cropped/Yemi Alade/Yemi Alade29.jpg',
      './data/cropped/Yemi Alade/Yemi Alade30.jpg',
      './data/cropped/Yemi Alade/Yemi Alade31.jpg',
      './data/cropped/Yemi Alade/Yemi Alade32.jpg',
      './data/cropped/Yemi Alade/Yemi Alade33.jpg',
      './data/cropped/Yemi Alade/Yemi Alade34.jpg',
      './data/cropped/Yemi Alade/Yemi Alade35.jpg',
      './data/cropped/Yemi Alade/Yemi Alade36.jpg',
      './data/cropped/Yemi Alade/Yemi Alade37.jpg',
      './data/cropped/Yemi Alade/Yemi Alade38.jpg',
      './data/cropped/Yemi Alade/Yemi Alade39.jpg',
      './data/cropped/Yemi Alade/Yemi Alade40.jpg',
      './data/cropped/Yemi Alade/Yemi Alade41.jpg',
      './data/cropped/Yemi Alade/Yemi Alade42.jpg',
      './data/cropped/Yemi Alade/Yemi Alade43.jpg',
      './data/cropped/Yemi Alade/Yemi Alade44.jpg',
      './data/cropped/Yemi Alade/Yemi Alade45.jpg',
      './data/cropped/Yemi Alade/Yemi Alade46.jpg',
      './data/cropped/Yemi Alade/Yemi Alade47.jpg',
      './data/cropped/Yemi Alade/Yemi Alade48.jpg',
      './data/cropped/Yemi Alade/Yemi Alade49.jpg']}




```python
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
class_dict
```




    {'Burna Boy': 0,
     'Davido': 1,
     'Falz': 2,
     'Tems': 3,
     'Tiwa Savage': 4,
     'Yemi Alade': 5}



### Data Preparation 
#### Splitting the wavelets transformed data into X and Y variables for training.


```python
x = []
y = []

for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(class_dict[celebrity_name])
```


```python
len(x[0])
```




    4096




```python
x = np.array(x).reshape(len(x), 4096).astype(float)
x.shape
```




    (181, 4096)




```python
x[0]
```




    array([135., 135., 135., ...,  68.,   1.,   0.])



### Training the Model.


```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```


```python
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(x_train, y_train)
pipe.score(x_test, y_test)
```




    0.782608695652174




```python
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
```


```python
model_params = {
    'svm' : {
        'model' : svm.SVC(gamma ='auto', probability =True),
        'params' : {
            'svc__C' : [1,10,100,1000],
            'svc__kernel' : ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model' : RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model' : LogisticRegression(solver='liblinear', multi_class='auto'),
        'params' : {
            'logisticregression__C' : [1,5,10]
        
        }    
    }
    
}     
                                          
```


```python
scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model' : algo,
        'best_score' : clf.best_score_,
        'best_params' : clf.best_params_    
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score', 'best_params'])
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>best_score</th>
      <th>best_params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>svm</td>
      <td>0.637037</td>
      <td>{'svc__C': 1, 'svc__kernel': 'linear'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>random_forest</td>
      <td>0.481481</td>
      <td>{'randomforestclassifier__n_estimators': 10}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>logistic_regression</td>
      <td>0.659259</td>
      <td>{'logisticregression__C': 1}</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_estimators['svm'].score(x_test,y_test)
```




    0.7608695652173914




```python
best_estimators['random_forest'].score(x_test,y_test)
```




    0.5434782608695652




```python
best_estimators['logistic_regression'].score(x_test,y_test)
```




    0.7608695652173914




```python
best_clf = best_estimators['svm']
```


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(x_test))
cm
```




    array([[ 4,  0,  0,  0,  1,  0],
           [ 1,  4,  0,  0,  0,  0],
           [ 0,  1,  7,  0,  0,  1],
           [ 0,  0,  0,  3,  0,  0],
           [ 1,  0,  0,  1,  5,  1],
           [ 0,  3,  0,  0,  1, 12]], dtype=int64)
