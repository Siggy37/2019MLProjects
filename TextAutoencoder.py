# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 00:04:13 2019

@author: brand
"""

import torch
import torch.nn as nn
import json

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_dim, latent_dim, num_layers)
        
    def forward(self, input):
        out, hidden = self.rnn(input)
        
        return out, hidden
        
    
class DecoderRNN(nn.Module):
    def __init__(self, decoder_dim, input_dim, latent_dim, num_layers):
        super(DecoderRNN, self).__init__()
        self.decoder_dim = decoder_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.linear1 = nn.Linear(decoder_dim, input_dim )
        self.rnn = nn.GRU(input_dim, input_dim, num_layers)

    def forward(self, input):
#        print(input)
        trans = self.linear1(input)
#        print(trans)
        out, hidden = self.rnn(trans)
        
        return out
        
class AE(nn.Module):
    def __init__(self, decoder_dim, input_dim, enc_latent_dim, dec_latent_dim, num_layers):
        super(AE, self).__init__()
        self.decoder_dim = decoder_dim
        self.input_dim = input_dim
        self.enc_latent_dim = enc_latent_dim
        self.dec_latent_dim = dec_latent_dim
        self.num_layers = num_layers
        self.encoder = EncoderRNN(input_dim, enc_latent_dim, num_layers)
        self.decoder = DecoderRNN(decoder_dim, input_dim, dec_latent_dim, num_layers)
        
    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded[0])
        return encoded, decoded                
        


def main(filepath, tensorname, epochs):    
    """
    returns a column vector representing a compressed space of the 
    information present in the sequential input
    """
    
    data = open(filepath).read()
    data = json.loads(data)
    data = torch.Tensor(data[tensorname])
    data = data.view(1, 49, 768).cuda()
    autoencoder = AE(1, 768, 1, 768, 1).cuda()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)    
    for i in range(epochs):
        encoded, decoded = autoencoder(data)
        optimizer.zero_grad()
        loss = loss_function(decoded, data)
        loss.backward()
        optimizer.step()
        
    return encoded[0]

if __name__ == '__main__':
    x = main('./bert/TextData/Training/TrainTensors_00HFP.json', 'question0',
             1000)
    print(x)
        
 