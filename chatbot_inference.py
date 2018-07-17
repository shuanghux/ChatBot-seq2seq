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
import string



def text_to_seq(text) :
    text = text.translate(translator)
    seq = []
    for word in text.lower().split():
        word_id = w2idx[word] if word in keys else 1
        seq.append(word_id)
    return seq


def seq_to_text(seq) :
    text = ''
    for word_id in seq :
        if word_id == end_id or word_id == pad_id :
            break;
        else :
            text += (idx2w[word_id]) + ' '
    return text

def build_feeder(model, text, go_id, end_id):
    seq = text_to_seq(text)
    seq_len = len(seq)
    seq = np.reshape(np.array(seq), (seq_len,1))
    return {
        model.encoder_inputs : seq,
        model.go_tokens: np.zeros(1) + go_id,
        model.end_token: end_id
    }



tf.reset_default_graph()
checkpoint_prefix = 'models'


metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')
xvocab_size = len(metadata['idx2w'])  
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

encoder_hidden_units = 1024
decoder_hidden_units = 2 * encoder_hidden_units
emb_dim = 300

keys = w2idx.keys()
translator = str.maketrans('', '', string.punctuation)


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
    print('Model successfully restored, chatbot ready to chat!')
else :
    sess.run(tf.global_variables_initializer())
    print('Unable to load trained model, will talk nonsense')


try:
	while True:
		text = input('Query   >  ')
		fd = build_feeder(model, text, start_id, end_id)
		out = sess.run(model.decoder_prediction, fd)
		response = seq_to_text(np.reshape(out,out.shape[0]).tolist())
		print       ('Respone >  ' + response)
except KeyboardInterrupt:
    print('\nChatting Session terminated!')

