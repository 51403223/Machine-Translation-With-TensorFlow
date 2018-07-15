import tensorflow as tf
from utils import embedding
import os
import numpy as np
import bleu

eos_vocab_id = 0
sos_vocab_id = 2
unk_vocab_id = 1


def create_dataset(sentences_as_ids):
    def generator():
        for sentence in sentences_as_ids:
            yield sentence

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int32)
    return dataset


def test_model():
    print('Loading word embeddings...')
    data_path = 'data/'  # path of data folder
    embeddingHandler = embedding.Embedding()

    ############### load embedding for source language ###############
    src_input_path = data_path + 'tst2012.vi'  # path to training file used for encoder
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
    tgt_input_path = data_path + 'tst2012.en'
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
    print('Creating dataset...')
    print('Number of test examples: ', training_size)

    # create training set for encoder (source)
    sentences_src_as_ids = embeddingHandler.convert_sentences_to_ids(dic_src, sentences_src)
    for sentence in sentences_src_as_ids:  # add <eos>
        sentence.append(eos_vocab_id)
    test_set_src = create_dataset(sentences_src_as_ids)
    test_set_src_len = create_dataset([[len(s)] for s in sentences_src_as_ids])

    # create training set for decoder (target)
    sentences_tgt_as_ids = embeddingHandler.convert_sentences_to_ids(dic_tgt, sentences_tgt)
    # for sentence_as_ids in sentences_tgt_as_ids:  # add </s> id to the end of each sentence of target language
    #     sentence_as_ids.append(eos_vocab_id)
    test_set_tgt = create_dataset(sentences_tgt_as_ids)
    test_set_tgt_len = create_dataset([[len(sentence)+1] for sentence in sentences_tgt_as_ids])
    # Note: [len(sentence)+1] for later <sos>/<eos>
    test_set_tgt_padding = create_dataset([np.ones(len(sentence)+1, np.float32) for sentence in sentences_tgt_as_ids])

    # create dataset contains both previous training sets
    train_dataset = tf.data.Dataset.zip((test_set_src, test_set_tgt, test_set_src_len, test_set_tgt_len, test_set_tgt_padding))
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
    # decode_seq_lens = tf.reshape(len_ys, shape=[batch_size])
    decode_seq_lens = encode_seq_lens * 2  # maximum iterations
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
    state_to_clone = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    decoder_initial_state = tf.contrib.seq2seq.AttentionWrapperState(
        cell_state=tf.nn.rnn_cell.LSTMStateTuple(
            c=tf.zeros_like(enc_2nd_states[-1].c, dtype=tf.float32),
            h=enc_2nd_states[-1].h
        ),
        attention=state_to_clone.attention,
        time=state_to_clone.time,
        alignments=state_to_clone.alignments,
        alignment_history=state_to_clone.alignment_history,
        attention_state=state_to_clone.attention_state
    )

    # projection
    tgt_vocab_size = len(vocab_tgt)
    weight_score = tf.Variable(
        tf.random_uniform(shape=[attention_output_size, tgt_vocab_size], minval=-0.1, maxval=0.1)
    )
    bias_score = tf.Variable(
        tf.zeros([batch_size, tgt_vocab_size])
    )

    # infer
    def loop_fn(time, cell_output, cell_state, loop_state):
        elements_finished = time >= decode_seq_lens  # finish by sentence length
        if cell_output is None:  # time = 0
            next_cell_state = decoder_initial_state
            next_input = tf.nn.embedding_lookup(embedding_tgt, [sos_vocab_id]*batch_size)
            emit_output = tf.constant(0)  # https://github.com/hanxiao/hanxiao.github.io/issues/8
        else:
            next_cell_state = cell_state
            score = tf.add(
                tf.matmul(cell_output, weight_score), bias_score
            )
            softmax = tf.nn.softmax(score)
            predict = tf.argmax(softmax, axis=-1, output_type=tf.int32)
            elements_finished = tf.logical_or(elements_finished, tf.equal(predict, eos_vocab_id))  # or finish by generated <eos>
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.nn.embedding_lookup(embedding_tgt, [eos_vocab_id]*batch_size),
                lambda: tf.nn.embedding_lookup(embedding_tgt, predict)
            )
            emit_output = predict

        next_loop_state = None
        return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    outputs_ta, final_state, _ = tf.nn.raw_rnn(attention_cell, loop_fn)
    outputs = outputs_ta.stack()  # [time, batch]
    outputs = tf.transpose(outputs)  # reshape to [batch, time]
    # -----------calculate score

    # dec_outputs_len = tf.shape(dec_outputs)[0]
    # def cond(i, *_):
    #     return tf.less(i, dec_outputs_len)
    # def body(i, _logits):
    #     score = tf.add(
    #         tf.matmul(dec_outputs[i], weight_score), bias_score
    #     )
    #     return i+1, _logits.write(i, score)
    # _, logits = tf.while_loop(
    #     cond, body, loop_vars=[0, tf.TensorArray(tf.float32, size=dec_outputs_len, clear_after_read=True)], swap_memory=True
    # )
    # labels = tf.transpose(tf.concat([y_batch, tf.reshape([eos_vocab_id]*batch_size, [batch_size, 1])], axis=-1))
    #

    #################### train ########################
    model_path = "./checkpoint_v2/model"
    checkpoint_path = "./checkpoint_v2"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        print('Inferring')
        sess.run(train_iter.initializer)
        references = []
        # Note: references has shape 3-d to pass into compute_bleu function
        # first dimension is batch size, second dimension is number of references for 1 translation
        # third dimension is length of each sentence (maybe differ from each other)
        translation = []
        while True:
        # for i in range(10):
        #     print(i)
            try:
                predictions, labels = sess.run([outputs, y_batch])
                # perform trimming <eos> to not to get additional bleu score by overlap padding
                predictions = [np.trim_zeros(predict, 'b') for predict in predictions]
                labels = [np.trim_zeros(lb, 'b') for lb in labels]
                # # convert ids to words
                # predictions = [embeddingHandler.ids_to_words(predict, vocab_tgt) for predict in predictions]
                # labels = [embeddingHandler.ids_to_words(lb, vocab_tgt) for lb in labels]
                references.extend(labels)
                translation.extend(predictions)
            except tf.errors.OutOfRangeError:
                break

        # compute bleu score
        reshaped_references = [[ref] for ref in references]
        bleu_score, *_ = bleu.compute_bleu(reshaped_references, translation, max_order=4, smooth=False)
        return bleu_score
