import numpy as np
import unicodedata, re
import time, os
from preprocessing import DatasetLoader
import tensorflow as tf
from Transformer import Transformer, translate
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
checkpoint_path = "saved_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model=Transformer()
print(latest)
model.load_weights(latest).expect_partial()
print("Source sentence: ", '大越史記外紀全書卷之')
print("Target sentence: ", 'Nhâm Tuất nguyên niên')
print("Predicted sentence: ", ' '.join(translate(model, '大越史記外紀全書卷之'.split(' '))))
print('________________________________________________________________________')
print("Source sentence: ", '君')
print("Target sentence: ", 'Lạc Long Quân')
print("Predicted sentence: ", ' '.join(translate(model, '貉龍君'.split(' '))))
print('________________________________________________________________________')
print("Source sentence: ", '諱崇')
print("Target sentence: ", 'Huý Sùng Lãm, Kinh Dương Vương chi tu dã')
print("Predicted sentence: ", ' '.join(translate(model, '諱崇'.split(' '))))
print('________________________________________________________________________')
print("Source sentence: ", '去')
print("Target sentence: ", 'đằng không nhi khứ')
print("Predicted sentence: ", ' '.join(translate(model, '去'.split(' '))))
print('________________________________________________________________________')
print("Source sentence: ", '蜀')
print("Target sentence: ", 'Thục Vương văn chi, cáo vương cầu vi hôn.')
print("Predicted sentence: ", ' '.join(translate(model, '蜀'.split(' '))))
