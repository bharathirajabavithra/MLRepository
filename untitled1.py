# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:07:38 2017

@author: bharathiraja
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
import text_helpers
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()


texts = []
with open("Sentences.txt", encoding="utf8") as f:
        for line in f:
            texts.append(line.rstrip('\r\n'))
            
# Start a graph session
sess = tf.Session()
texts=sorted(texts)

# Declare model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 480
 #2000
generations = 500
model_learning_rate = 0.5

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = 2       # How many words to consider left and right.

# Add checkpoints to training
save_embeddings_every = 500
print_valid_every = 100
print_loss_every = 100

# Declare stop words
stops = stopwords.words('english')

print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)

texts = [x for x in texts if len(x.split()) > 2]

# Build our data set and dictionaries
print('Creating Dictionary')
def build_dataset(words, vocabulary_size):
  count = [['RARE', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reverse_dictionary

def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)

word_dictionary, word_dictionary_rev = build_dataset(texts, vocabulary_size)

#word_dictionary = build_dataset(texts, vocabulary_size)
#word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)


#valid_words = ['cannot', 'goods', 'invoice', 'delivery', 'create', 'job']
valid_words=[]
valid_examples=np.random.choice(250, 250, replace=False)
for x in valid_examples:
    valid_words.append(word_dictionary_rev[x])
# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]  
graph = tf.Graph()
with graph.as_default():
    print('Creating Model')
    # Define Embeddings:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    
    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                   stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Create data/target placeholders
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
    y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    
    # Lookup the word embedding
    # Add together window embeddings:
    embed = tf.zeros([batch_size, embedding_size])
    for element in range(2*window_size):
        embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])
    
    # Get loss from prediction
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=y_target,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
                                         
    # Create optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
    
    # Cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # Create model saving operation
    saver = tf.train.Saver({"embeddings": embeddings})
    
    #Add variable initializer.
    init = tf.global_variables_initializer()
with tf.Session(graph=graph) as sess:
    #sess.run(init)
    init.run()
    
    # Filter out sentences that aren't long enough:
    text_data = [x for x in text_data if len(x)>=(2*window_size+1)]
    
    
    # Run the CBOW model.
    print('Starting Training')
    loss_vec = []
    loss_x_vec = []
    WriteOuput = open('Output.txt','w')
    average_loss = 0
    for i in range(generations):
        batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,
                                                                      window_size, method='cbow')
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}
    
        # Run the train step
        #sess.run(optimizer, feed_dict=feed_dict)
#        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
#        average_loss += loss_val
#        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
#        average_loss += loss_val
        # Return the loss
        if (i+1) % print_loss_every == 0:
            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            #average_loss /= print_loss_every
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            #print('Loss at step {} : {}'.format((i+1), average_loss))
            average_loss=0
          
        # Validation: Print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_words)):
                valid_word = word_dictionary_rev[valid_examples[j]]
                top_k = 5 # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k+1]
                log_str = "Nearest to {}:".format(valid_word)
                WriteOuput.write("Nearest to {}:".format(valid_word))
                WriteOuput.write('\n')
                for k in range(top_k):
                    #print(nearest[k])
                    close_word = word_dictionary_rev[nearest[k]]
                    log_str = '{} {},' .format(log_str, close_word)
                    WriteOuput.write('\t{}'.format(close_word))
                    WriteOuput.write('\n')
               # print(log_str)
    
        # Save dictionary + embeddings
        if (i+1) % save_embeddings_every == 0:
            # Save vocabulary dictionary
            with open(os.path.join('vocab.pkl'), 'wb') as f:
                pickle.dump(word_dictionary, f)
            
            # Save embeddings
            model_checkpoint_path = os.path.join(os.getcwd(),'cbow_embeddings.ckpt')
            save_path = saver.save(sess, model_checkpoint_path)
            print('Model saved in file: {}'.format(save_path))
    final_embeddings = normalized_embeddings.eval()
    WriteOuput.close()
        