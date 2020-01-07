# -*- coding=utf-8 -*-
from tensorflow.keras import layers
from tensorflow.keras import models

class BasicBlock(layers.Layer):
    '''基本残差块层1  resnet18,resnet34
    conv(3,3)->conv(3,3)
    '''
    def __init__(self,filter_num,stride=1):
        super(BasicBlock,self).__init__()
        # conv1
        self.conv1=layers.Conv2D(filters=filter_num,kernel_size=(3,3),
                                 strides=stride,padding='same')
        self.bn1=layers.BatchNormalization()
        self.relu1=layers.Activation(activation='relu')

        # conv2
        self.conv2 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3),
                                   strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        # shortcut
        if stride!=1:
            self.downsample=layers.Conv2D(filter_num,kernel_size=(1,1),strides=stride)
        else:
            self.downsample=lambda x:x
        self.relu_out=layers.Activation('relu')

    def call(self, inputs, training=None):
        x=self.conv1(inputs)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.downsample(inputs)+x
        return self.relu_out(x)

class BasicBlock_NIN(layers.Layer):
    '''基本残差块层2  resnet50,resnet101,resnet152...
    conv(1,1)->conv(3,3)->conv(1,1)
    '''
    def __init__(self, filter_num, stride=1):
        super(BasicBlock_NIN, self).__init__()
        # conv1
        self.nin_conv1 = layers.Conv2D(filters=filter_num, kernel_size=(1, 1),
                                   strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation(activation='relu')

        # conv2
        self.conv2 = layers.Conv2D(filters=filter_num, kernel_size=(3, 3),
                                   strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation(activation='relu')

        # conv3
        self.nin_conv3 = layers.Conv2D(filters=filter_num*4, kernel_size=(1, 1),
                                       strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()

        # shortcut
        if stride != 1:
            self.downsample = layers.Conv2D(filter_num*4, kernel_size=(1, 1), strides=stride)
        else:
            self.downsample = layers.Conv2D(filter_num*4, kernel_size=(1, 1), strides=1)
        self.relu_out = layers.Activation('relu')

    def call(self, inputs, training=None):
        x=self.nin_conv1(inputs)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x = self.relu2(x)

        x = self.nin_conv3(x)
        x = self.bn3(x)
        x=self.downsample(inputs)+x
        return self.relu_out(x)

class ResNet(models.Model):
    def __init__(self,layer_dims,num_classes=10,basicblock_nin=False):
        super(ResNet, self).__init__()
        self.stem=models.Sequential(
            [layers.Conv2D(64,(7,7),strides=(2,2),padding='same'),
             layers.BatchNormalization(),
             layers.Activation('relu'),
             layers.MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same')]
        )
        self.block1=self._build_resnet_blocks(64,layer_dims[0],nin=basicblock_nin)
        self.block2=self._build_resnet_blocks(128,layer_dims[1],stride=2,nin=basicblock_nin)
        self.block3=self._build_resnet_blocks(256,layer_dims[2],stride=2,nin=basicblock_nin)
        self.block4=self._build_resnet_blocks(512,layer_dims[3],stride=2,nin=basicblock_nin)

        self.avg_layer=layers.GlobalAveragePooling2D()
        self.fc=layers.Dense(num_classes)

    def _build_resnet_blocks(self,filter_nums,blocks,stride=1,nin=False):
        if nin:
            basicblock=BasicBlock_NIN
        else:
            basicblock = BasicBlock
        blk=models.Sequential()
        blk.add(basicblock(filter_nums,stride=stride))
        for _ in range(blocks-1):
            blk.add(basicblock(filter_nums,stride=1))
        return blk

    def call(self, inputs, training=None, mask=None):
        x=self.stem(inputs)
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.avg_layer(x)
        x=self.fc(x)
        return x


def ResNet18(num_classes):
    return ResNet([2,2,2,2],num_classes)

def ResNet34(num_classes):
    return ResNet([3,4,6,3],num_classes)

def ResNet50(num_classes):
    return ResNet([3,4,6,3],num_classes,basicblock_nin=True)

def ResNet101(num_classes):
    return ResNet([3,4,23,3],num_classes,basicblock_nin=True)

def ResNet152(num_classes):
    return ResNet([3,8,36,3],num_classes,basicblock_nin=True)



