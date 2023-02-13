# -*- coding: utf-8 -*-
import tensorflow as tf

class Block(tf.keras.Model):
    def __init__(self, embed = 1024, dropout_rate = 0.3):
        super(Block, self).__init__()
        self.d1 = tf.keras.layers.Dense(embed)
        self.bn1= tf.keras.layers.BatchNormalization()
        self.ac1= tf.keras.layers.ReLU()
        self.dp1= tf.keras.layers.Dropout(dropout_rate)

    def call(self,x):
        x = self.dp1(self.ac1(self.bn1(self.d1(x))))
        return x

class OuterBlock(tf.keras.Model):
    def __init__(self,embed = 1024, dropout_rate = 0.3):
        super(OuterBlock, self).__init__()
        self.fc = tf.keras.layers.Dense(embed)
        self.b1 = Block(embed, dropout_rate)
        self.b2 = Block(embed, dropout_rate)
        
    def call(self, inputs):
        skip = self.fc(inputs)
        x = self.b1(inputs)
        x = self.b2(x)
        return skip + x

class LinearModel(tf.keras.Model):
    def __init__(self, embed = 1024, dropout_rate = 0.3):
        super(LinearModel,  self).__init__()
        self.b1 = OuterBlock(embed, dropout_rate)
        self.b2 = OuterBlock(embed, dropout_rate)
        self.d3 = tf.keras.layers.Dense(156)
        
    def call(self, inputs):
        x = self.b1(inputs)
        x = self.b2(x)
        x = self.d3(x)
        return x
