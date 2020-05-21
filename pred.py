from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model=load_model('gen200.h5')
print(model.summary())

x=np.random.random_sample((100,)).reshape(1,-1)

img=model.predict(x)
plt.imshow(img.reshape(128,128,3))
plt.show()