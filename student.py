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

"""
############################# QUESTION ###################################
Briefly describe how your program works, and explain any design and training
decisions you made along the way.

The algorithm chosen to solve the classification and text sentiment is a Bidirectional LSTM. 
Single LSTM was also implemented but was outperfomed by the ladder.
Bi-LSTM was chosen for its flexibility to take different input length and have the ability to "remember" previous
seen data situations that would suggest the type of text sentiment or class. CNN was considered but since 
the maximum number of words in a document was unknown RNN was prefered. 

Some simple preprocessing and tokenization was applied using standard libraries. No postprocessing was implemented.
The tokenizer chosen was basic english, other tokenizers were implemented but the basic one performed better.
A combination of different hyperparameters and methods were implemented such as
    - Different number of layers 
    - different glove dimensions
    - number of hidden nodes 
    - number of epochs
    - different preprocessing stage
    - optimizer (SGD vs ADAM)

The best performance for the number of layers was 2 anything greater or lower decreased the performance.
Increasing the number of Glove dimensions to 300 increased the accuracy by 3% but not surprisingly increased the time to train by twice as much.
Increasing the number of hidden nodes increased the final accuracy by 1% - 2% on average. Other values 10, 32, 54 and 100 were considered but 54 outperformed all the others.
The number of epochs was found to decrease the final accuracy and time of training so the default value of 10 was always considered with other hyperparameters.
Different values were implemented 10, 20, 30, 40,50.

A simple preprocessing approach was implemented, removing non - basic english characters. A dictionary with a big collection of stopWords was tested.
However, it removed some critical information and affected the performance of the network. After looking at the data after tokenizing some custom preprocessing was applied
to increase the performance, such as removing lengthy words, fixing some words such as dont, didnt, have and are but there were no significant results.

For the optimizer simple SGD was first implemented. However, it was slower and less accurate than the Adam optimizer.

##############################################################################
"""

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
    def reduce_lengthening(text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    
    processed = []
    for text in sample:
        # remove punctuation
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)
        text = re.sub('ve', 'have', text)
        text = re.sub('ll', 'all', text)
        text = re.sub('re', 'are', text)
        text = re.sub(r'(\bdon\b|\bdidn\b)', 'do not', text)
        text = reduce_lengthening(text)
        if text == 'do not':
            processed.append('do')
            text = 'not'
        processed.append(text)

    return processed

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

#stopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
stopWords ={'t','s'}
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

    # round to 1 or 0 and change the size to actuall output size
    ratingOut = torch.round(ratingOutput) 
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
        #lstm layer define hyperparameters
        embedding_dim = 300
        hidden_dim = 40
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
        # dense layer rating/sentiment
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
        #dense multiclass output
        dense_outputs_multi=self.fc_m(hidden)
        #dense rating/sentiment output
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
        category_loss = self.entropy(categoryOutput,categoryTarget.long())
        rating_loss = self.bce(ratingOutput,one_hot_rating)
        #The loss is a combination of the rating and multiclass loss
        return category_loss + rating_loss

       
        

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.01) #optimizer changed to adam
