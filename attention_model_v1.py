import tensorflow as tf
from utils import embedding
import os
import time
import numpy as np

eos_vocab_id = 0
sos_vocab_id = 2
unk_vocab_id = 1


def create_dataset(sentences_as_ids):
    def generator():
        for sentence in sentences_as_ids:
            yield sentence

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int32)
    return dataset


def train_model():
    print('Loading word embeddings...')
    data_path = 'data/'  # path of data folder
    embeddingHandler = embedding.Embedding()

    ############### load embedding for source language ###############
    src_input_path = data_path + 'train.vi'  # path to training file used for encoder
    src_embedding_output_path = data_path + 'embedding.vi'  # path to file word embedding
    src_vocab_path = data_path + 'vocab.vi'  # path to file vocabulary

    vocab_src, dic_src = embeddingHandler.load_vocab(src_vocab_path)
    sentences_src = embeddingHandler.load_sentences(src_input_path)
    if not os.path.exists(src_embedding_output_path):
        word2vec_src = embeddingHandler.create_embedding(sentences_src, vocab_src, src_embedding_output_path)
    else:
        word2vec_src = embeddingHandler.load_embedding(src_embedding_output_path)
    embedding_src = embeddingHandler.parse_embedding_to_list_from_vocab(word2vec_src, vocab_src)
    embedding_src = tf.constant(embedding_src)

    ################ load embedding for target language ####################
    tgt_input_path = data_path + 'train.en'
    tgt_embedding_output_path = data_path + 'embedding.en'
    tgt_vocab_path = data_path + 'vocab.en'

    vocab_tgt, dic_tgt = embeddingHandler.load_vocab(tgt_vocab_path)
    sentences_tgt = embeddingHandler.load_sentences(tgt_input_path)
    if not os.path.exists(tgt_embedding_output_path):
        word2vec_tgt = embeddingHandler.create_embedding(sentences_tgt, vocab_tgt, tgt_embedding_output_path)
    else:
        word2vec_tgt = embeddingHandler.load_embedding(tgt_embedding_output_path)
    embedding_tgt = embeddingHandler.parse_embedding_to_list_from_vocab(word2vec_tgt, vocab_tgt)
    embedding_tgt = tf.constant(embedding_tgt)

    if word2vec_src.vector_size != word2vec_tgt.vector_size:
        print('Word2Vec dimension not equal')
        exit(1)
    if len(sentences_src) != len(sentences_tgt):
        print('Source and Target data not match number of lines')
        exit(1)
    word2vec_dim = word2vec_src.vector_size  # dimension of a vector of word
    training_size = len(sentences_src)
    print('Word2Vec dimension: ', word2vec_dim)
    print('-------------------------------')

    ################## create dataset ######################
    batch_size = 64
    num_epochs = 12
    print('Creating dataset...')
    print('Number of training examples: ', training_size)

    # create training set for encoder (source)
    sentences_src_as_ids = embeddingHandler.convert_sentences_to_ids(dic_src, sentences_src)
    for sentence in sentences_src_as_ids:  # add <eos>
        sentence.append(eos_vocab_id)
    train_set_src = create_dataset(sentences_src_as_ids)
    train_set_src_len = create_dataset([[len(s)] for s in sentences_src_as_ids])

    # create training set for decoder (target)
    sentences_tgt_as_ids = embeddingHandler.convert_sentences_to_ids(dic_tgt, sentences_tgt)
    # for sentence_as_ids in sentences_tgt_as_ids:  # add </s> id to the end of each sentence of target language
    #     sentence_as_ids.append(eos_vocab_id)
    train_set_tgt = create_dataset(sentences_tgt_as_ids)
    train_set_tgt_len = create_dataset([[len(sentence)+1] for sentence in sentences_tgt_as_ids])
    # Note: [len(sentence)+1] for later <sos>/<eos>
    train_set_tgt_padding = create_dataset([np.ones(len(sentence)+1, np.float32) for sentence in sentences_tgt_as_ids])
    ## padding matrix
    # target_weights = create_dataset([np.ones(len(sentence) + 1) for sentence in sentences_tgt_as_ids])

    # create dataset contains both previous training sets
    train_dataset = tf.data.Dataset.zip((train_set_src, train_set_tgt, train_set_src_len, train_set_tgt_len, train_set_tgt_padding))
    train_dataset = train_dataset.shuffle(buffer_size=training_size, seed=9)
    # train_dataset = train_dataset.shuffle(buffer_size=training_size)
    train_dataset = train_dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(batch_size, ([None], [None], [1], [1], [None])))
    train_iter = train_dataset.make_initializable_iterator()
    x_batch, y_batch, len_xs, len_ys, padding_mask = train_iter.get_next()
    # Note: len_xs and len_ys have shape [batch_size, 1]
    print('-------------------------------')
    #################### build graph ##########################
    hidden_size = word2vec_dim  # number of hidden unit
    print('Building graph...')
    encode_seq_lens = tf.reshape(len_xs, shape=[batch_size])
    # ---------encoder first layer
    enc_1st_outputs, enc_1st_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
        cell_bw=tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
        inputs=tf.nn.embedding_lookup(embedding_src, x_batch),
        sequence_length=encode_seq_lens,
        swap_memory=True,
        time_major=False,
        dtype=tf.float32
    )  # [batch, time, hid]
    fw_enc_1st_hid_states, bw_enc_1st_hid_states = enc_1st_outputs
    # fw_enc_1st_last_hid, bw_enc_1st_last_hid = enc_1st_states

    # ----------encoder second layer
    num_layers = 2
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(hidden_size*2)] * num_layers
    )
    enc_2nd_outputs, enc_2nd_states = tf.nn.dynamic_rnn(
        cell=stacked_lstm,
        inputs=tf.concat([fw_enc_1st_hid_states, bw_enc_1st_hid_states], axis=-1),
        sequence_length=encode_seq_lens,
        dtype=tf.float32,
        swap_memory=True,
        time_major=False
    )

    # ----------decoder
    encode_output_size = hidden_size*2
    decode_seq_lens = tf.reshape(len_ys, shape=[batch_size])
    attention_output_size = 256
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=encode_output_size,
        memory=enc_2nd_outputs,  # require [batch, time, ...]
        memory_sequence_length=encode_seq_lens,
        dtype=tf.float32
    )
    attention_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=encode_output_size)
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        attention_cell, attention_mechanism,
        attention_layer_size=attention_output_size
    )
    add_sos = tf.concat([tf.reshape([sos_vocab_id]*batch_size, [batch_size, 1]), y_batch], axis=-1)
    decoder_initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    decoder_initial_state = decoder_initial_state.clone(cell_state=enc_2nd_states[-1])
    dec_outputs, _ = tf.nn.dynamic_rnn(
        cell=attention_cell,
        inputs=tf.nn.embedding_lookup(embedding_tgt, tf.transpose(add_sos)),
        initial_state=decoder_initial_state,
        sequence_length=decode_seq_lens,
        dtype=tf.float32,
        swap_memory=True,
        time_major=True
    )

    # -----------calculate score
    tgt_vocab_size = len(vocab_tgt)
    weight_score = tf.Variable(
        tf.random_uniform(shape=[attention_output_size, tgt_vocab_size], minval=-0.1, maxval=0.1)
    )
    bias_score = tf.Variable(
        tf.zeros([batch_size, tgt_vocab_size])
    )
    dec_outputs_len = tf.shape(dec_outputs)[0]
    def cond(i, *_):
        return tf.less(i, dec_outputs_len)
    def body(i, _logits):
        score = tf.add(
            tf.matmul(dec_outputs[i], weight_score), bias_score
        )
        return i+1, _logits.write(i, score)
    _, logits = tf.while_loop(
        cond, body, loop_vars=[0, tf.TensorArray(tf.float32, size=dec_outputs_len, clear_after_read=True)], swap_memory=True
    )
    labels = tf.transpose(tf.concat([y_batch, tf.reshape([eos_vocab_id]*batch_size, [batch_size, 1])], axis=-1))

    # ----------loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits.stack())  # [time,batch]
    apply_penalty = cross_entropy * tf.transpose(tf.cast(padding_mask, tf.float32))
    loss = tf.reduce_sum(apply_penalty) / batch_size

    # ----------optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)  # derivation of loss by params
    max_gradient_norm = 5
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    starting_rate = 1.0
    decay_epochs = 4  # decay learning rate on every n epochs exclude first n epochs
    decay_step = (training_size // batch_size) * decay_epochs  # num_step_in_single_epoch * n
    learning_rate = tf.train.exponential_decay(learning_rate=starting_rate, global_step=global_step,
                                               decay_steps=decay_step, decay_rate=0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    #################### train ########################
    log_frequency = 100
    model_path = "./checkpoint_framework/model"
    checkpoint_path = "./checkpoint_framework"
    loss_epochs = tf.TensorArray(tf.float32, size=num_epochs, dynamic_size=True)
    training_epoch = tf.Variable(0, trainable=False, name='training_epoch')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
            print('...............Restored from checkpoint_framework')
        except:
            sess.run(tf.global_variables_initializer())
        start_epoch = sess.run(training_epoch)
        for epoch in range(start_epoch, num_epochs):
            print('Training epoch', epoch + 1)
            start_time = time.time()
            total_loss = 0
            sess.run(train_iter.initializer)
            while True:
                try:
                    _, l, lr, step = sess.run([optimizer, loss, learning_rate, global_step])
                    total_loss += l
                    if np.isnan(l):
                        return False
                    print('Step {0}: loss={1} lr={2}'.format(step, l, lr))
                    # if step % log_frequency == 0:
                    #     print('Step {0}: loss={1} lr={2}'.format(step, l, lr))
                except tf.errors.OutOfRangeError:
                    avg_loss = total_loss / (training_size // batch_size)
                    loss_epochs = loss_epochs.write(epoch, tf.cast(avg_loss, tf.float32))  # write average loss of epoch
                    sess.run(training_epoch.assign(epoch + 1))  # starting epoch if restore
                    saver.save(sess, model_path, epoch)
                    print('Average loss=', avg_loss)
                    break

            print('Epoch {} train in {} minutes'.format(epoch + 1, (time.time() - start_time) / 60.0))
            print('------------------------------------')

        loss_summary = sess.run(loss_epochs.stack())
        loss_epochs.close()
        np.savetxt(checkpoint_path + '/loss_summary.txt', loss_summary, fmt='%10.5f')
        return True

train_model()
# # train loop prevents 'nan' occurs
# while True:
#     train_result = train_model()
#     if train_result is True:
#         break
