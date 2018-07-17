import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from tensorflow.python.layers.core import Dense
from tensorlayer.layers import *
import tensorlayer as tl

class chatbot_seq2seq(object):
    '''
    chatbot implemented with seq2seq model.
    Encoder: Bi-LSTM
    Decoder: LSTM-Attention
    '''
    
    #define init function for model
    def __init__(self, vocab_size, input_embedding_size, encoder_hidden_units, decoder_hidden_units):
        
        '''Initialize Hparams'''
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.use_attention = False
        
        #Define placeholder
        self.build_placeholder()
        
        #Define embedding matrix
        self.build_emb_matrix()
        
        #Define Encoder as Bidirectional LSTM Cell
        self.encoder_cell, self.encoder_final_state = self.build_encoder()
        
        #Define Decoder as Basic LSTM Cell
        self.build_decoder()
        
        #Define Loss and Prediction
        self.build_op()

    
    
    
    def build_placeholder(self) :
        '''Initialize Placeholders'''
        #inputs dimension [encoder_max_time, batch_size]
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        #retrieve_seq_length need argument with shape [batch_size, max_seq_len]
        #encoder_inputs_length with shape [batch_size]
        self.encoder_inputs_length = retrieve_seq_length_op2(tf.transpose(self.encoder_inputs))
        #decoder_inputs with shape [max_seq_len, batch_size], in the form [start_id, how, are, you, pad_id...]
        self.decoder_inputs = tf.placeholder(shape = (None, None), dtype = tf.int32, name = 'decoder_inputs')
        #decoder targets with shape [max_seq_len, batch_size], in the form [how, are, you, end_id, pad_id ...]
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        #target_masks [max_seq_len, batch_size]
        self.decoder_masks = tf.placeholder(shape = (None, None), dtype = tf.int32, name = 'decoder_masks')
        #go_tokens used for inference decode, with shape [batch_size]
        self.go_tokens = tf.placeholder(shape = (None,), dtype = tf.int32, name = 'go_tokens')
        #EOS token for inference, scalar
        self.end_token = tf.placeholder(shape = (), dtype = tf.int32, name = 'end_token')
        
    def build_encoder(self):
        encoder_cell = LSTMCell(self.encoder_hidden_units)

        ((encoder_fw_outputs,
          encoder_bw_outputs),
         (encoder_fw_final_state,
          encoder_bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell,
                cell_bw=encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32, time_major=True)
            )

        #Concatenates tensors along one dimension.
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        self.encoder_outputs = encoder_outputs
        #letters h and c are commonly used to denote "output value" and "cell state". 
        #http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
        #Those tensors represent combined internal state of the cell, and should be passed together. 

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        #TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
        encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
        
        return encoder_cell, encoder_final_state
    
    def build_emb_matrix(self) :
        '''Initialize Embedding Matrix'''
        #this operation is moved to CPU as some bugs in emb_lookup's GPU implementation
        with tf.device("/cpu:0"):
            #randomly initialized embedding matrrix that can fit input sequence
            #used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
            #reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
            self.encoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1, 1), dtype=tf.float32)
            self.decoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1, 1), dtype=tf.float32)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.encoder_inputs)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs)
            
    
    def build_decoder(self) :
        '''Define Decoder as Basic LSTM Cell'''
        self.decoder_cell = LSTMCell(self.decoder_hidden_units)
        if self.use_attention is True:
            attn_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = self.encoder_hidden_units, 
                memory = attn_states,
                memory_sequence_length = self.encoder_inputs_length
            )
            self.attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = self.decoder_cell,
                attention_mechanism = attn_mechanism,
                attention_layer_size = self.encoder_hidden_units,
                name = 'attn_decoder_cell'
            )
            #define decoder initial state, retrieve from encoder_final_state
            self.train_decoder_init_state = self.attn_decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state = self.encoder_final_state)
        
        self.encoder_max_time, self.batch_size = tf.unstack(tf.shape(self.encoder_inputs))
        #????????????
        self.decoder_lengths = self.encoder_inputs_length * 0 + tf.reduce_max(retrieve_seq_length_op2(tf.transpose(self.decoder_masks)))
        # +2 additional steps, +1 leading <EOS> token for decoder inputs
        self.projection_layer = Dense(units = self.vocab_size, use_bias = True)
        #Training Helper
        self.train_helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_inputs_embedded, self.decoder_lengths, time_major = True)
        
        #Training Decoder
        self.train_decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, self.train_helper, self.encoder_final_state, output_layer = self.projection_layer)
        #Training Output
        self.train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = self.train_decoder,output_time_major = True, maximum_iterations= 26)
        #Inferencing Helper
        self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.decoder_embeddings, start_tokens = self.go_tokens, end_token = self.end_token)
         #Training Decoder
        self.infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, self.infer_helper, self.encoder_final_state, output_layer = self.projection_layer)
        #Training Output
        self.infer_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = self.infer_decoder,output_time_major = True, maximum_iterations= 25)

        


        #Beam Search
        self.beam_encoder_state = tf.contrib.seq2seq.tile_batch(
            self.encoder_final_state, multiplier = 5)
        self.beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = self.decoder_embeddings,
            start_tokens = self.go_tokens,
            end_token = self.end_token,
            initial_state = self.beam_encoder_state,
            beam_width = 5,
            output_layer = self.projection_layer)
        self.beam_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = self.beam_decoder,output_time_major = True, maximum_iterations= 25)
    def build_op(self) :
        self.train_logits = self.train_decoder_outputs.rnn_output
        self.decoder_prediction = tf.argmax(self.infer_decoder_outputs.rnn_output, 2)
        #cross_entropy with shape [max_seq_len, batch_size]
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
                logits = self.train_logits)

        #loss function
        self.loss = tf.reduce_sum(self.cross_entropy * tf.cast(self.decoder_masks, tf.float32))
        #train it 
        self.train_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(self.loss)