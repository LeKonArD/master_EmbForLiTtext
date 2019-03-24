import pandas as pd
import random
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, LSTM, Concatenate, concatenate,Reshape, multiply, GRU, Permute, merge, Bidirectional, Multiply, Lambda, RepeatVector
from keras.models import Model
from keras.metrics import categorical_accuracy
import keras.backend as K
from keras.utils import plot_model
from keras import regularizers
from keras.optimizers import Adadelta,Nadam,SGD,Adam
import keras
from keras.engine.topology import Layer
from keras import initializers
import fastText as ft

## fill theses with paths
traindata = ""
testdata = ""


model_ft = ft.load_model("/home/ext/konle/master/dnb_embeddings.bin")

def get_activations_keras(model, inputs, layer_name):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations

def generate_embeddings_eval(batch_size):
  
  x_batch = []
  y_batch = []
  while True:
   
    data = pd.read_csv(testdata, sep="\t")
    data = data.sample(frac=1, random_state=200)
    for index, row in data.iterrows():
        my_sent = list(row[1:-1].astype(str))
        t_embed = [model_ft.get_word_vector(w) for w in my_sent]
        if np.shape(t_embed) != (200,300):
           continue
            
        x_batch.append(t_embed)  
          
          
        if row["label"] == "liebes":
          y_batch.append([1,0,0,0])
        if row["label"] == "horror":
          y_batch.append([0,1,0,0])
        if row["label"] == "krimi":
          y_batch.append([0,0,1,0])
        if row["label"] == "scifi":
          y_batch.append([0,0,0,1])

          
        if len(y_batch) == batch_size:
       
          
          if np.shape(x_batch) != (batch_size,200,300):
            x_batch = []
            y_batch = []
            continue
          
          yield np.array(x_batch), np.array(y_batch)

          x_batch = []
          y_batch = []

def generate_embeddings(batch_size):
  
  x_batch = []
  y_batch = []
  while True:
   
    data = pd.read_csv(traindata, sep="\t")
    data = data.sample(frac=1, random_state=200)
    for index, row in data.iterrows():
        
        
        my_sent = list(row[1:-1].astype(str))
        
        t_embed = [model_ft.get_word_vector(w) for w in my_sent]
        if np.shape(t_embed) != (200,300):
          continue
            
        x_batch.append(t_embed)
          
          
        if row["label"] == "liebes":
          y_batch.append([1,0,0,0])
        if row["label"] == "horror":
          y_batch.append([0,1,0,0])
        if row["label"] == "krimi":
          y_batch.append([0,0,1,0])
        if row["label"] == "scifi":
          y_batch.append([0,0,0,1])

          
        if len(y_batch) == batch_size:
       
          
          if np.shape(x_batch) != (batch_size,200,300):
            x_batch = []
            y_batch = []
            continue
          
          yield np.array(x_batch), np.array(y_batch)

          x_batch = []
          y_batch = []

def generate_sentences(batch_size):
  
  x_batch = []
  y_batch = []
  while True:
   
    data = pd.read_csv(testdata, sep="\t",error_bad_lines=False,  quotechar="%")
    data = data.sample(frac=1, random_state=200)
    for index, row in data.iterrows():
        
        if row["label"] == "liebes":
          y_batch.append([1,0,0,0])
        if row["label"] == "horror":
          y_batch.append([0,1,0,0])
        if row["label"] == "krimi":
          y_batch.append([0,0,1,0])
        if row["label"] == "scifi":
          y_batch.append([0,0,0,1])


        my_sent = list(row[1:-1].astype(str))
        
        sentence = flair.data.Sentence(" ".join(my_sent))
        t_embed = []
        for token in sentence:
          t_embed.append(token.text) 
          
        x_batch.append(t_embed)  
          
          
        if len(y_batch) == batch_size:
          
          if np.shape(x_batch) != (batch_size,200):
            x_batch = []
            y_batch = []
            continue
            
          yield np.array(x_batch), np.array(y_batch)

          x_batch = []
          y_batch = []

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Attention(Layer):
	def __init__(self, regularizer=None, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.regularizer = regularizer
		self.supports_masking = True

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.context = self.add_weight(name='context', 
									   shape=(input_shape[-1], 1),
									   initializer=initializers.RandomNormal(
									   		mean=0.0, stddev=0.05, seed=None),
									   regularizer=self.regularizer,
									   trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
		attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)

		if mask is not None:
			# use only the inputs specified by the mask
			# import pdb; pdb.set_trace()
			attention = attention*K.cast(mask, 'float32')

		weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
		return weighted_sum

	def compute_output_shape(self, input_shape):
		print(input_shape)
		return (input_shape[0], input_shape[-1])

def declare_model():
  ## Input
  sample = Input(batch_shape=(100, 200, 300))
  
  ## Word Encoder
  lstm_out = Bidirectional(GRU(128, return_sequences=True))(sample)
  ## Hidden Representation 
  dense_transform_w = Dense(200, activation='relu', name='dense_transform_w')(lstm_out)
  ## Attention
  attention_weighted_text = Attention(name='sentence_attention')(dense_transform_w)
  ## Prediction
  predictions = Dense(4, activation='softmax')(attention_weighted_text)

  model = Model(inputs=sample, outputs=[predictions])
  model.compile(optimizer=Adam(lr=1e-3, clipnorm=4),
              loss='categorical_crossentropy',
              metrics=[f1])
  print(model.summary())
  return model


m = declare_model()

history = m.fit_generator(generate_embeddings(100), steps_per_epoch=240, epochs=10)
with open("fasttext_local_history","w") as file:
  file.write(str(history.history))


score = m.evaluate_generator(generate_embeddings_eval(100), steps=80)
print(score)
m.save("model_ft_local.h5")
