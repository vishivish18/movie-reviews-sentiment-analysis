import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

train = pd.read_csv("./all/labeledTrainData.tsv", delimiter="\t")
test = pd.read_csv("./all/testData.tsv", delimiter="\t")

print(train)
print (test)


def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"   ", " ", text) # Remove any extra spaces
    text = re.sub(r"  ", " ", text)
    
    # Return a list of words
    return(text)


train_clean = []
for review in train.review:
    train_clean.append(clean_text(review))


test_clean = []
for review in test.review:
    test_clean.append(clean_text(review))


# Tokenize the reviews
all_reviews = train_clean + test_clean
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("Fitting is complete.")

train_seq = tokenizer.texts_to_sequences(train_clean)
print("train_seq is complete.")

test_seq = tokenizer.texts_to_sequences(test_clean)
print("test_seq is complete")

word_index = tokenizer.word_index

print(train_seq[0])


max_review_length = 200

train_pad = pad_sequences(train_seq, maxlen = max_review_length)
print("train_pad is complete.")

test_pad = pad_sequences(test_seq, maxlen = max_review_length)
print("test_pad is complete.")



x_train, x_valid, y_train, y_valid = train_test_split(train_pad, train.sentiment, test_size = 0.15, random_state = 2)



def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
def get_test_batches(x, batch_size):
    '''Create the batches for the testing data'''
    n_batches = len(x)//batch_size
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size]





def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers, dropout, learning_rate, multiple_fc, fc_units):
    '''Build the Recurrent Neural Network'''

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Create the embeddings
    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words, 
                                    embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                         output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    
    # Set the initial state
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(
                                        cell,         
                                        embed,
                                        initial_state=initial_state)    
    
    # Create the fully connected layers
    with tf.name_scope("fully_connected"):
        
        # Initialize the weights and biases
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        
        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                    num_outputs = fc_units,
                    activation_fn = tf.sigmoid,
                    weights_initializer = weights,
                    biases_initializer = biases)
        
        dense = tf.contrib.layers.dropout(dense, keep_prob)
        
        # Depending on the iteration, use a second fully connected layer
        if multiple_fc == True:
            dense = tf.contrib.layers.fully_connected(dense,
                        num_outputs = fc_units,
                        activation_fn = tf.sigmoid,
                        weights_initializer = weights,
                        biases_initializer = biases)
            
            dense = tf.contrib.layers.dropout(dense, keep_prob)
    
    # Make the predictions
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense, 
                          num_outputs = 1, 
                          activation_fn=tf.sigmoid,
                          weights_initializer = weights,
                          biases_initializer = biases)
        
        tf.summary.histogram('predictions', predictions)
    
    # Calculate the cost
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    
    # Train the model
    with tf.name_scope('train'):    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Determine the accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), 
                                        tf.int32), 
                                        labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'keep_prob','initial_state',        
                    'final_state','accuracy', 'predictions', 'cost', 
                    'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph


tf.reset_default_graph()