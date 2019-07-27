# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:03:58 2019

@author: brand
"""
import tensorflow as tf
import readtext as rt
import databaser as db
from keras import models
from keras.layers import Dense, Dropout, BatchNormalization, Bidirectional
from keras.layers import LSTM
from keras.models import Sequential
import numpy as np
import extract_features as ef
from keras.layers import CuDNNLSTM, TimeDistributed
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from databaser import max_characters
from keras.metrics import sparse_categorical_accuracy
BATCH_SIZE = 0
FEATURE_LEN = 768
SEQ_LEN = 1

#max_label_sequence = max_entities * 2

class train_and_predict:
    def __init__(self, sequence_length=0):
        self.sequence_length = sequence_length


    """
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    """



    def train_model(self, embeddings, labels, seq_len):
    #    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        SEQ_LEN = seq_len
        BATCH_SIZE = len(embeddings)//167
        model = Sequential()
        
        label_len = labels.tolist()
        label_len = len(label_len[0])
        
        embeddings = embeddings.reshape((len(embeddings), SEQ_LEN, FEATURE_LEN))
        labels = labels.reshape((len(embeddings), 1, label_len))
        labels = labels[:, 0,:] 
        print(embeddings)
        print(labels)
        model.add(CuDNNLSTM(128, input_shape=(SEQ_LEN, FEATURE_LEN), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())    
        
        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())    
        
        model.add(Dense(label_len, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(embeddings, labels, epochs=100, batch_size=BATCH_SIZE, verbose=2)
        model.save("Entity_Rec3.model")
    
    def train_bidirectional(self, embeddings, labels, seq_len):
        SEQ_LEN = seq_len
        BATCH_SIZE = len(embeddings)//167
        
        label_len = labels.tolist()
        label_len = len(label_len[0])
        
        embeddings = embeddings.reshape((len(embeddings), SEQ_LEN, FEATURE_LEN))
        labels = labels.reshape((len(embeddings), 1, label_len))
        labels = labels[:, 0,:] 
        print(embeddings)
        print(labels)
        
        model = Sequential()
    
        model.add(Bidirectional(LSTM(128, input_shape=(SEQ_LEN, FEATURE_LEN), dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=False)))
        model.add(BatchNormalization())
    #    model.add(Attention(10))

        model.add(Dense(label_len, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy']
        )    
        model.fit(embeddings, labels,
              batch_size=BATCH_SIZE,
              epochs=100)
        model.save("SparseEntity.model")
        
        
        
    def predict(self, model, tensor):
    #    model = models.load_model("BidEntity_Red.model")
         seq_len = int(open('sequencesave.txt').read())
         line_len = len(tensor[0][0])
         pad = np.zeros(line_len)
         tensor = tensor[0]
         pad_len = seq_len - len(tensor)
         print(pad_len)
         tensor = tensor.tolist()
         pad = pad.tolist()
         for i in range(pad_len):
             tensor.append(pad)
         tensor = np.array(tensor)
         tensor = tensor.reshape((1, seq_len, len(tensor[0])))
         print(tensor)
         prediction = model.predict(tensor)
         return prediction
         


if __name__ == '__main__':
    program = train_and_predict()
    i = 0
    choice = input('Train [0], Predict [1]: ')
    if choice == '0':
        database_file = 'engtrain.bio.txt'
        outfile = 'outfile1.txt'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
    #    print(tf.test.gpu_device_name())   
        sentences = rt.read_file(database_file)
        rt.build_and_dump(sentences, outfile)    
        sents = db.read_file('outfile1.txt')
        tensors = ef.build_tensors()
    #    print(tensors)
        embeddings, labels, seq_len = db.find_entities(sents)
        print(labels)
        seqwrite = open('sequencesave.txt', mode='w')
        seqwrite.write(str(seq_len))
        seqwrite.close()
    #    print(embeddings)
    
     #   program.train_model(embeddings, labels, seq_len)
        
        
        program.train_bidirectional(embeddings, labels, seq_len)
        
    elif choice == '1':
        model = models.load_model("SparseEntity.model")
        sentence = input('Input: ')
        sentence = sentence.replace("\n", "")
        writef = open('predictionfile.txt', mode='w')
        writef.write(sentence)
        writef.close()
        infile = open('predictionfile.txt').read()
        print(infile)
        ef.main('predictionfile.txt')
        tensor, sl = ef.build_tensors()
#        print(tensor)
        prediction = program.predict(model, tensor)
        print(prediction)

        prediction = prediction[0].tolist()
        print(max(prediction))
        print(prediction.index(max(prediction)))
        
        
    
    
    
    