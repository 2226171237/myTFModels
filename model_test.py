# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.keras import datasets

from models import ResNet18


gpus=tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)  # 根据运算需要设置占有显存大小
    except RuntimeError as e:
        print(e)

resnet50=ResNet18(10)
resnet50.build(input_shape=(2,224,224,3))
imgs=tf.random.normal(shape=(64,224,224,3))

import time
start=time.time()
y=resnet50(imgs)
print(time.time()-start)

