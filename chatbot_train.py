import numpy as np #matrix math 
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from tensorflow.python.layers.core import Dense
from datasets.twitter import data
import data_utils
from tensorlayer.layers import *
import tensorlayer as tl
import time, sys
from chatbot_model import chatbot_seq2seq
import matplotlib.pyplot as plt

tf.reset_default_graph()
checkpoint_prefix = 'models'

# dataset 1. 'twitter', 2. 'cornell_corpus'
dataset = sys.argv[1]
num_epoch = int(sys.argv[2])
# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/' + dataset + '/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# pre-process training data part I
trainX = trainX.tolist()
trainY = trainY.tolist()

# converge test
# trainX = trainX[0:32*5]
# trainY = trainY[0:32*5]



trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)

# parameters 
xseq_len = len(trainX)
yseq_len = len(trainY)
assert xseq_len == yseq_len
BATCH_SIZE = 512
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
encoder_hidden_units = 1024
decoder_hidden_units = 2 * encoder_hidden_units
emb_dim = 300

encoder_max_time = 25
decoder_max_time = encoder_max_time

# updata parameters with preprocessing
w2idx = metadata['w2idx']
idx2w = metadata['idx2w']
unk_id = w2idx['unk']
pad_id = w2idx['_']
start_id = xvocab_size
end_id = xvocab_size+1
w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
idx2w = idx2w + ['start_id', 'end_id']
xvocab_size = yvocab_size = xvocab_size + 2
# A data for Seq2Seq should look like this:
# input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
# decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
# target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
# target_mask : [1, 1, 1, 1, 0]

def chatbot_train_next_batch(model, iterator, go_id, end_id):
    X, Y = iterator.__next__()
    #[batch_size, max_seq_len]
    _encoder_seqs = tl.prepro.pad_sequences(X)
    _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id = end_id)
    ##[batch_size, max_seq_len]
    _target_seqs = tl.prepro.pad_sequences(_target_seqs)
    _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id = start_id, remove_last = False)
    #[batch_size, max_seq_len]
    _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
    #[batch_size, max_seq_len]
    _target_masks = tl.prepro.sequences_get_mask(_target_seqs)
    return {
        model.encoder_inputs: np.array(_encoder_seqs).T,
        model.decoder_inputs: np.array(_decode_seqs).T,
        model.decoder_targets: np.array(_target_seqs).T,
        model.decoder_masks: _target_masks.T,
        model.go_tokens: np.zeros(len(_encoder_seqs)) + go_id,
        model.end_token: end_id
    }

model = chatbot_seq2seq(
    vocab_size = xvocab_size, 
    input_embedding_size = emb_dim, 
    encoder_hidden_units = encoder_hidden_units, 
    decoder_hidden_units = decoder_hidden_units
)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,device_count={'GPU':1}))
saver = tf.train.Saver()
if tf.train.checkpoint_exists(checkpoint_prefix) :
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_prefix))
    print('Restoring model parameters successful')
else :
    sess.run(tf.global_variables_initializer())
    print('Model initialized')



batchs_per_epoch = int(len(trainX) / BATCH_SIZE)
batches_cost = []
for i in range(num_epoch):
    save_path = saver.save(sess, checkpoint_prefix + '/twitter_model.ckpt')
    iterator = tl.iterate.minibatches(inputs = trainX, targets = trainY, batch_size = BATCH_SIZE, shuffle = False)
    time_start = time.time()
    for j in range(batchs_per_epoch):
        fd = chatbot_train_next_batch(model, iterator, start_id, end_id)
        _, l = sess.run([model.train_op, model.loss], fd)
        batches_cost.append(l);
        speed = (j+1)/(time.time() - time_start)
        sys.stdout.write('\rBatch: %i/%i --Step: %i/%i -- loss: %.10f -- Speed: %.3f batches/second' % (i+1, num_epoch, j+1, batchs_per_epoch, l, speed))
        sys.stdout.flush()
        
save_path = saver.save(sess, checkpoint_prefix + '/twitter_model.ckpt')
plt.plot(batches_cost)
plt.show()
np.save('loss_atten.npy', np.array(batches_cost))