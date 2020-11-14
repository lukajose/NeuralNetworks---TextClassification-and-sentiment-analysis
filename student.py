#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

# import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import re
import torch
from torchtext.data.utils import get_tokenizer
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################




def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    tokenizer = get_tokenizer("basic_english")
    processed = tokenizer(sample)
    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    
    processed = []
    for text in sample:
        # remove punctuation
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)
        processed.append(text)
    return processed

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # convert output label to actual
    ratingOut = torch.round(ratingOutput) # round to 1 or 0
    b,_ = ratingOutput.size()
    ratingOut = ratingOut.view(b).long()

    #convert back to categorical int class
    pred_label = [int(torch.argmax(i)) for i in categoryOutput]
    pred_label = torch.Tensor(pred_label).long()
    return ratingOut, pred_label

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        #lstm layer
        embedding_dim = 300
        hidden_dim = 54
        output_dim_multiclass = 5
        output_dim_sentiment = 1
        n_layers = 2
        bidirectional = True
        dropout = 0.2
        self.lstm = tnn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #dense layer multiclass
        self.fc_m = tnn.Linear(hidden_dim * 2, output_dim_multiclass)
        # dense layer sentiment analysis
        self.fc_s = tnn.Linear(hidden_dim * 2, output_dim_sentiment)

        #activation function
        self.act = tnn.Sigmoid()


    def forward(self, feed, length):

        # Multiclass classification
        packed_embedded = tnn.utils.rnn.pack_padded_sequence(feed, length,batch_first=True)
        _, (hidden, _) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        #dense multi
        dense_outputs_multi=self.fc_m(hidden)
        #dense sentiment
        dense_outputs_senti=self.fc_s(hidden)
        #Final activation function multiclass
        categoryOutput=self.act(dense_outputs_multi)
        #Final output activation Sentiment analysis
        ratingOutput= self.act(dense_outputs_senti)
        return (ratingOutput,categoryOutput)
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.bce = tnn.BCELoss()
        self.entropy = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):

        #encode multiclass
        one_hot_rating = ratingTarget.float().view(ratingOutput.size())
        #category_loss = self.entropy(categoryOutput,categoryTarget)
        category_loss = self.entropy(categoryOutput,categoryTarget.long())
        rating_loss = self.bce(ratingOutput,one_hot_rating)
        
        return category_loss + rating_loss

       
        

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01)
