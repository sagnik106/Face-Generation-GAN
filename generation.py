import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
import cv2

gpus=tf.config.experimental.get_visible_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model=tf.keras.models.load_model("models/gen15400.h5", compile=False)

while True:
    try:
        n=int(input("Generate image (press any number -1 to exit) : "))
    except:
        print("Wrong input")
        continue
    if n==-1:
        break
    latent=np.random.random(size=(1,500))
    p=model.predict(latent)
    img=p[0].reshape(128,128,3)
    cv2.imshow('IMAGE',img)
    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()
    n=input("Save the image (in current directory)?(y/n) : ")
    if n.lower() == 'y':
        cv2.imwrite("%d.jpg"%time.time(), img)