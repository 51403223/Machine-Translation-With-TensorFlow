import tensorflow as tf
from utils import embedding
import numpy as np
from beam_search import raw_rnn_for_beam_search
from beam_search import extract_from_tree
from beam_search import get_word_ids
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MachineTranslator:
    def __init__(self, beam_width=1):
        with tf.Graph().as_default():
            eos_vocab_id = 0
            sos_vocab_id = 2
            unk_vocab_id = 1

            sentence = tf.placeholder(tf.int32)
            model_path = 'checkpoint_v1/model-11'
            self.beam_width = beam_width

            data_path = 'data/'  # path of data folder
            embeddingHandler = embedding.Embedding()

            ############### load embedding for source language ###############
            src_embedding_output_path = data_path + 'embedding.vi'  # path to file word embedding
            src_vocab_path = data_path + 'vocab.vi'  # path to file vocabulary

            vocab_src, dic_src = embeddingHandler.load_vocab(src_vocab_path)
            word2vec_src = embeddingHandler.load_embedding(src_embedding_output_path)
            embedding_src = embeddingHandler.parse_embedding_to_list_from_vocab(word2vec_src, vocab_src)
            embedding_src = tf.constant(embedding_src)

            ################ load embedding for target language ####################
            tgt_embedding_output_path = data_path + 'embedding.en'
            tgt_vocab_path = data_path + 'vocab.en'
            vocab_tgt, dic_tgt = embeddingHandler.load_vocab(tgt_vocab_path)
            word2vec_tgt = embeddingHandler.load_embedding(tgt_embedding_output_path)
            embedding_tgt = embeddingHandler.parse_embedding_to_list_from_vocab(word2vec_tgt, vocab_tgt)
            embedding_tgt = tf.constant(embedding_tgt)

            word2vec_dim = word2vec_src.vector_size  # dimension of a vector of word

            ################## create dataset ######################
            batch_size = 64
            sentence = tf.concat([[sos_vocab_id], sentence], axis=0)
            x_batch = tf.gather([sentence], [0] * batch_size)  # duplicate sentence into a batch, shape [batch, len]
            len_sentence = tf.shape(sentence)[-1]
            #################### build graph ##########################
            hidden_size = word2vec_dim  # number of hidden unit
            encode_seq_lens = tf.convert_to_tensor([len_sentence] * batch_size)
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

            # ----------encoder second layer
            num_layers = 2
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(hidden_size * 2)] * num_layers
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
            encode_output_size = hidden_size * 2
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
            decoder_initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            decoder_initial_state = decoder_initial_state.clone(cell_state=enc_2nd_states[-1])

            # projection
            tgt_vocab_size = len(vocab_tgt)
            weight_score = tf.Variable(
                tf.random_uniform(shape=[attention_output_size, tgt_vocab_size], minval=-0.1, maxval=0.1)
            )
            bias_score = tf.Variable(
                tf.zeros([batch_size, tgt_vocab_size])
            )

            # beam search
            def loop_fn(time, cell_output, cell_state, log_probs, beam_finished):
                elements_finished = time >= decode_seq_lens  # finish by sentence length
                if cell_output is None:  # initialize step
                    next_cell_state = tuple(decoder_initial_state for _ in range(beam_width))
                    next_input = tuple(
                        tf.nn.embedding_lookup(embedding_tgt, [sos_vocab_id] * batch_size) for _ in range(beam_width))
                    predicted_ids = tf.convert_to_tensor(
                        [0] * beam_width)  # https://github.com/hanxiao/hanxiao.github.io/issues/8
                    new_log_probs = tf.zeros([batch_size, beam_width])
                    new_beam_finished = tf.fill([batch_size, beam_width], value=False)
                    parent_indexs = None
                else:
                    def not_time_0():
                        next_cell_state = cell_state
                        # find predicted_ids
                        values_list = []
                        indices_list = []
                        for i in range(beam_width):
                            score = tf.add(
                                tf.matmul(cell_output[i], weight_score), bias_score
                            )
                            softmax = tf.nn.softmax(score)
                            log_prob = tf.log(softmax)
                            values, indices = tf.nn.top_k(log_prob, beam_width,
                                                          sorted=True)  # [batch, beam], [batch, beam]
                            # Note: indices is ids of words as well
                            values = tf.add(values, tf.expand_dims(log_probs[:, i], -1))  # sum with previous log_prob
                            values_list.append(values)
                            indices_list.append(indices)
                        concat_vlist = tf.concat(tf.unstack(values_list, axis=0),
                                                 axis=-1)  # [batch_size, beam_width*beam_width]
                        concat_ilist = tf.concat(tf.unstack(indices_list, axis=0), axis=-1)
                        top_values, index_in_vlist = tf.nn.top_k(concat_vlist, beam_width,
                                                                 sorted=True)  # [batch_size, beam_width]
                        # Note: in tf.nn.top_k, if sorted=False then it's values will be SORTED ASCENDING

                        predicted_ids = get_word_ids(index_in_vlist, concat_ilist, batch_size)
                        predicted_ids = tf.stack(predicted_ids)  # [batch_size, beam_width]

                        # new_beam_finished = tf.logical_or(tf.equal(predicted_ids, eos_vocab_id), beam_finished)

                        # find parent_ids that match word_ids_to_add
                        parent_indexs = index_in_vlist // beam_width
                        # find new_log_probs
                        new_log_probs = top_values

                        # shift top-k according to beam_finished
                        # which means we will shift predicted_ids, new_log_probs, parent_indexs
                        def shift(tensor_1D, num_shift, vacancy_value):
                            """
                            shift from left to right
                            """
                            shift_value = tensor_1D[:beam_width - num_shift]
                            fill_vacancy = tf.fill([num_shift], vacancy_value)
                            return tf.concat([fill_vacancy, shift_value], axis=0)

                        ids_arr = []
                        probs_arr = []
                        parents_arr = []
                        num_shifts = tf.map_fn(lambda beam: tf.reduce_sum(tf.cast(beam, tf.int32)),
                                               beam_finished, dtype=tf.int32)
                        # Note: we don't shift using new_beam_finished to avoid newly finish
                        # which will update -inf to final_log_probs
                        for i in range(batch_size):
                            num_shift = num_shifts[i]
                            ids_arr.append(shift(predicted_ids[i], num_shift, eos_vocab_id))
                            probs_arr.append(shift(new_log_probs[i], num_shift, -np.inf))
                            parents_arr.append(shift(parent_indexs[i], num_shift, -1))
                        valid_shape = tf.shape(beam_finished)
                        predicted_ids = tf.stack(ids_arr)
                        predicted_ids = tf.reshape(predicted_ids, valid_shape)
                        new_log_probs = tf.stack(probs_arr)
                        new_log_probs = tf.reshape(new_log_probs, valid_shape)
                        parent_indexs = tf.stack(parents_arr)
                        parent_indexs = tf.reshape(parent_indexs, valid_shape)

                        new_beam_finished = tf.logical_or(tf.equal(predicted_ids, eos_vocab_id), beam_finished)

                        # define next_input
                        finished = tf.reduce_all(elements_finished)
                        next_input = tuple(
                            tf.cond(
                                finished,
                                lambda: tf.nn.embedding_lookup(embedding_tgt, [eos_vocab_id] * batch_size),
                                lambda: tf.nn.embedding_lookup(embedding_tgt, predicted_ids[:, i])
                            ) for i in range(beam_width)
                        )

                        return elements_finished, next_input, next_cell_state, predicted_ids, new_log_probs, new_beam_finished, parent_indexs

                    def time_0():
                        next_cell_state = cell_state
                        # find next_input
                        score = tf.add(
                            tf.matmul(cell_output[0], weight_score), bias_score
                        )
                        softmax = tf.nn.softmax(score)
                        log_prob = tf.log(softmax)
                        top_values, predicted_ids = tf.nn.top_k(log_prob, beam_width,
                                                                sorted=True)  # [batch_size, beam_width]

                        new_beam_finished = beam_finished

                        parent_indexs = tf.fill([batch_size, beam_width], value=-1)

                        new_log_probs = top_values

                        finished = tf.reduce_all(elements_finished)
                        next_input = tuple(
                            tf.cond(
                                finished,
                                lambda: tf.nn.embedding_lookup(embedding_tgt, [eos_vocab_id] * batch_size),
                                lambda: tf.nn.embedding_lookup(embedding_tgt, predicted_ids[:, i])
                            ) for i in range(beam_width)
                        )

                        return elements_finished, next_input, next_cell_state, predicted_ids, new_log_probs, new_beam_finished, parent_indexs

                    # Important note: we won't feed <sos> at step 0 because it will lead to all same results on all beams
                    # instead, we feed top-k predictions generated from feeding <sos> as input
                    # other returns will be pass without change
                    elements_finished, next_input, next_cell_state, predicted_ids, new_log_probs, new_beam_finished, parent_indexs = tf.cond(
                        tf.equal(time, 0), time_0, not_time_0)

                return elements_finished, next_input, next_cell_state, predicted_ids, new_log_probs, new_beam_finished, parent_indexs

            predicted_ids_ta, parent_ids_ta, penalty_lengths, final_log_probs = raw_rnn_for_beam_search(attention_cell,
                                                                                                        loop_fn)
            translation_ta = extract_from_tree(predicted_ids_ta, parent_ids_ta, batch_size, beam_width)
            outputs = translation_ta.stack()  # [time, batch, beam]
            # choose best translation with maximum sum log probability
            normalize_log_probs = final_log_probs / penalty_lengths
            index = tf.argmax(tf.reshape(normalize_log_probs, shape=[-1]), output_type=tf.int32)
            transpose_outputs = tf.transpose(outputs, perm=[1, 2, 0])  # transpose to [batch, beam, time]
            batch_index = index // beam_width
            beam_index = index % beam_width
            final_output = transpose_outputs[batch_index, beam_index, :]

            #################### infer ########################
            saver = tf.train.Saver()
            sess = tf.Session()
            saver.restore(sess, model_path)

            self.sess = sess
            self.final_output = final_output
            self.sentence = sentence
            self.embeddingHandler = embeddingHandler
            self.dic_src = dic_src
            self.vocab_tgt = vocab_tgt

    # translate
    def translate(self, user_input):
        user_input = user_input.split()
        translation_original = self.sess.run(self.final_output,
                            feed_dict={self.sentence: self.embeddingHandler.words_to_ids(user_input, self.dic_src)})
        translation_trimmed_eos = np.trim_zeros(translation_original, 'b')
        output_translation = self.embeddingHandler.ids_to_words(translation_trimmed_eos, self.vocab_tgt)
        return " ".join(output_translation)
