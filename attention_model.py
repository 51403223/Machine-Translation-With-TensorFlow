import tensorflow as tf
from utils import embedding
import os
import time
import numpy as np

eos_vocab_id = 0
sos_vocab_id = 2
unk_vocab_id = 1

# this version use predefined, not trainable word embedding
# attention is gain from first layer of encoder
class LSTMcell():
    def __init__(self, hidden_size, input_size, batch_size):
        """
        :param hidden_size: dimension of hidden unit
        :param input_size: dimension of word2vec
        :param batch_size: number of sentences in a mini-batch
        """
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # link: https://www.coursera.org/learn/nlp-sequence-models/lecture/ftkzt/recurrent-neural-network-model
        # Note: gamma_gate = sigmoid([a,x] * W + b)
        # 'a' of shape[batch_size, hidden_size]
        # 'x' of shape[batch_size, input_size]
        # [a,x] has shape[batch_size, hidden_size + input_size]
        xavier_tanh = np.sqrt(1.0 / (hidden_size + input_size))
        xavier_sigmoid = np.sqrt(4.0 / (hidden_size + input_size))
        self.weight_update = tf.Variable(
            tf.random_uniform(shape=[hidden_size + input_size, hidden_size],
                              minval=-0.1, maxval=0.1, seed=1) * xavier_sigmoid
        )
        self.weight_forget = tf.Variable(
            tf.random_uniform(shape=[hidden_size + input_size, hidden_size],
                              minval=-0.1, maxval=0.1, seed=2) * xavier_sigmoid
        )
        self.weight_candidate = tf.Variable(
            tf.random_uniform(shape=[hidden_size + input_size, hidden_size],
                              minval=-0.1, maxval=0.1, seed=3) * xavier_tanh
        )
        self.weight_output = tf.Variable(
            tf.random_uniform(shape=[hidden_size + input_size, hidden_size],
                              minval=-0.1, maxval=0.1, seed=4) * xavier_sigmoid
        )

        self.bias_update = tf.Variable(tf.zeros(shape=[batch_size, hidden_size]))
        # self.bias_forget = tf.Variable(tf.random_normal(shape=[batch_size, hidden_size], seed=2))
        self.bias_forget = tf.Variable(tf.ones(shape=[batch_size, hidden_size]))
        self.bias_candidate = tf.Variable(tf.zeros(shape=[batch_size, hidden_size]))
        self.bias_output = tf.Variable(tf.zeros(shape=[batch_size, hidden_size]))

    def run_step(self, x, c, a):
        """
        Run step t
        :param x: input at time step t (in this case, batch of word2vec)
        :param c: cell state from previous step, aka c<t-1>
        :param a: hidden state from previous step, aka a<t-1>
        :return: tuple of tensors (new_cell_state, new_hidden_state)
        """
        concat_matrix = tf.concat(
            [a, x], axis=1,
            # name='concatenate'
        )
        # Note: shape[0] of matrix will be (hidden_size + input_dimension)
        # --> tf.concat([a, x], axis=1) and not else

        # new cell state candidate
        candidate = tf.tanh(
            tf.matmul(concat_matrix, self.weight_candidate) + self.bias_candidate,
            # name='create_candidate'
        )

        # forget gate
        gamma_f = tf.sigmoid(
            tf.matmul(concat_matrix, self.weight_forget) + self.bias_forget,
            # name='forget_gate'
        )

        # update gate
        gamma_u = tf.sigmoid(
            tf.matmul(concat_matrix, self.weight_update) + self.bias_update,
            # name='update_gate'
        )

        # output gate
        gamma_o = tf.sigmoid(
            tf.matmul(concat_matrix, self.weight_output) + self.bias_output,
            # name='output_gate'
        )

        # compute cell state at step t (this step)
        c_new = tf.add(
            x=tf.multiply(gamma_u, candidate), y=tf.multiply(gamma_f, c),
            # name='c_t'
        )

        # compute hidden state at step t
        a_new = tf.multiply(
            x=gamma_o, y=tf.tanh(c_new),
            # name='a_t'
        )

        return c_new, a_new


class EncoderBasic:
    """
    Uni-direction Encoder with a single LSTM cell
    """

    def __init__(self, lstm_cell, embeddings):
        self.lstm_cell = lstm_cell
        self.embeddings = embeddings  # embeddding list of word2vec

    def encode(self, batch_of_sentences):
        """
        Encode the sentence to vector represent
        :param batch_of_sentences: batch of source sentences (in this case, each sentence is a list of word indices)
        :return:
        """
        cell_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        hidden_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        sentence_length = tf.shape(batch_of_sentences)[1]  # length of a sentence
        hidden_states = tf.TensorArray(tf.float32, size=sentence_length, dynamic_size=True, clear_after_read=False)

        def cond(i, *_):
            return tf.less(i, sentence_length)

        def body(i, c, hid_states):
            x = batch_of_sentences[:, i]
            # x = tf.map_fn(lambda e: self.embeddings[e], x, dtype=tf.float32)  # transform to batch of vector
            # x = tf.map_fn(lambda e: tf.nn.embedding_lookup(self.embeddings, e),
            #               x, dtype=tf.float32)  # transform to batch of vector
            x = tf.nn.embedding_lookup(self.embeddings, x)
            c, new_hs = tf.cond(tf.equal(i, 0),
                                true_fn=lambda: self.lstm_cell.run_step(x, c, hidden_state),
                                false_fn=lambda: self.lstm_cell.run_step(x, c, hid_states.read(i - 1)))
            hid_states = hid_states.write(i, new_hs)
            return i + 1, c, hid_states

        _, _, hidden_states = tf.while_loop(cond, body, [0, cell_state, hidden_states])
        hidden_states_stack = hidden_states.stack()
        hidden_states.close()
        return hidden_states_stack

    def extract_more_info(self, batch_of_input):
        """
        Encode the sentence to vector represent
        :param batch_of_sentences: batch of source sentences (in this case, each sentence is a list of word indices)
        :return:
        """
        cell_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        hidden_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        sentence_length = tf.shape(batch_of_input)[1]  # length of a sentence
        hidden_states = tf.TensorArray(tf.float32, size=sentence_length, dynamic_size=True, clear_after_read=False)

        def cond(i, *_):
            return tf.less(i, sentence_length)

        def body(i, c, hid_states):
            x = batch_of_input[:, i]
            c, new_hs = tf.cond(tf.equal(i, 0),
                                true_fn=lambda: self.lstm_cell.run_step(x, c, hidden_state),
                                false_fn=lambda: self.lstm_cell.run_step(x, c, hid_states.read(i - 1)))
            hid_states = hid_states.write(i, new_hs)
            return i + 1, c, hid_states

        _, _, hidden_states = tf.while_loop(cond, body, [0, cell_state, hidden_states])
        hidden_states_stack = hidden_states.stack()
        hidden_states.close()
        return hidden_states_stack


class BidirectionalEncoder:
    def __init__(self, embeddings, fw_cell, bw_cell):
        self.fw_encoder = EncoderBasic(fw_cell, embeddings)
        self.bw_encoder = EncoderBasic(bw_cell, embeddings)

    def encode(self, batch_of_senteces):
        """
        :param batch_of_senteces:
        :return: concatenated hidden states of forward and backward cell, shape is [time, batch, hidden_size * 2]
        """
        batch_of_senteces_reverse = tf.reverse(batch_of_senteces, axis=[1])
        fw_hid_states = self.fw_encoder.encode(batch_of_senteces)  # shape [time, batch, hidden_size]
        bw_hid_states = self.bw_encoder.encode(batch_of_senteces_reverse)
        # concatenate hidden states
        concat_hid_states = tf.concat([fw_hid_states, bw_hid_states], axis=2)
        return concat_hid_states


class GlobalAttentionDecoder:
    def __init__(self, lstm_cell, vocab_size, embeddings, attention_input_size, attention_output_size):
        self.lstm_cell = lstm_cell
        self.weight_score = tf.Variable(
            tf.random_uniform(shape=[lstm_cell.hidden_size, vocab_size], minval=-0.1, maxval=0.1, seed=5)
        )
        self.bias_score = tf.Variable(tf.zeros([lstm_cell.batch_size, vocab_size]))
        self.embeddings = embeddings
        self.weight_attention = tf.Variable(
            tf.random_uniform([attention_input_size, attention_output_size], minval=-0.1, maxval=0.1, seed=6)
        )

    def decode(self, labels, encode_hid_states, last_encode_hid_state):
        """
        Decode the represent vector from encoder (in this case, it's last_encode_hid_state).
        Using global attention as defined in "Effective Approaches to Attention-based Neural Machine
        Translation" by Minh Thang Luong
        :param labels: batch of label sentences
        whose shape is [batch_size, sentence_length] and don't need to add ending with <eos>
        :param encode_hid_states: hidden states produced by encoder, used for attention
        :param last_encode_hid_state: last hidden state produced by encoder
        :return:
        """
        cell_state = tf.zeros([self.lstm_cell.batch_size, self.lstm_cell.hidden_size])
        sentence_length = tf.shape(labels)[1]
        hidden_states = tf.TensorArray(tf.float32, size=sentence_length + 1, dynamic_size=True, clear_after_read=False)
        logits = tf.TensorArray(tf.float32, size=sentence_length + 1, dynamic_size=True,
                                clear_after_read=False)  # model's predictions
        labels_transform = tf.TensorArray(tf.int32, size=sentence_length + 1, dynamic_size=True,
                                          clear_after_read=False)  # labels converted shape to match logits
        # feed <sos> to generate first word
        cell_state, hidden_state = self.lstm_cell.run_step(
            [self.embeddings[sos_vocab_id]] * self.lstm_cell.batch_size,
            cell_state, last_encode_hid_state)
        hidden_states = hidden_states.write(0, hidden_state)
        score_vector = tf.add(
            tf.matmul(hidden_state, self.weight_score), self.bias_score
        )
        # don't add softmax here because of later using tf.softmax_cross_entropy_v2
        logits = logits.write(0, score_vector)

        def cond(i, *_):
            return tf.less_equal(i, sentence_length)

        def body(i, c, predicts, hid_states, lbs_transform):
            y = labels[:, i - 1]  # input shift left by 1
            lbs_transform = lbs_transform.write(i - 1, y)  # also shift by 1
            # y = tf.map_fn(lambda e: tf.nn.embedding_lookup(self.embeddings, e), y, dtype=tf.float32)
            y = tf.nn.embedding_lookup(self.embeddings, y)
            c, new_hs = self.lstm_cell.run_step(y, c, hid_states.read(i - 1))  # predict next word
            # attention
            context_vector = self.context_vector(encode_hid_states, new_hs)
            new_hs = tf.tanh(
                tf.matmul(tf.concat([context_vector, new_hs], axis=1), self.weight_attention)
            )
            hid_states = hid_states.write(i, new_hs)
            score = tf.add(
                tf.matmul(new_hs, self.weight_score), self.bias_score
            )
            predicts = predicts.write(i, score)
            return i + 1, c, predicts, hid_states, lbs_transform

        _, _, logits, hidden_states, labels_transform = tf.while_loop(cond, body,
                                                                      [1, cell_state, logits, hidden_states,
                                                                       labels_transform])  # loop at time step 0+1
        # add <eos> to label
        labels_transform = labels_transform.write(sentence_length, [eos_vocab_id] * self.lstm_cell.batch_size)

        logits_stack = logits.stack()
        logits.close()
        hidden_states_stack = hidden_states.stack()
        hidden_states.close()
        labels_transform_stack = labels_transform.stack()
        labels_transform.close()
        return logits_stack, hidden_states_stack, labels_transform_stack

    def context_vector(self, encode_hid_states, hidden_state):
        """
        create context vector at time step t
        :param encode_hid_states: hidden states of source sentences, shape [time, batch, hidden_size]
        :param hidden_state: current hidden state of decoder, shape [batch, hidden_size]
        :return: context vector with shape [batch, hidden_size]
        """
        score_attention = tf.reduce_sum(hidden_state * encode_hid_states, axis=-1)  # shape [time, batch]
        alpha = tf.nn.softmax(score_attention)
        alpha_to_3d = tf.reshape(alpha, shape=[tf.shape(alpha)[0], tf.shape(alpha)[1], 1])
        context_vector = tf.reduce_sum(alpha_to_3d * encode_hid_states, axis=0)
        return context_vector
        #a=tf.map_fn(lambda e:tf.reduce_sum(h1*e, axis=-1), s1, dtype=tf.float32)
        #b=tf.reduce_sum(h1*s1, axis=-1)



def create_dataset(sentences_as_ids):
    def generator():
        for sentence in sentences_as_ids:
            yield sentence

    dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int32)
    return dataset


# translate
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
        word2vec_src = embeddingHandler.create_embedding(sentences_src, vocab_src)
        embeddingHandler.save_embedding(word2vec_src, src_embedding_output_path)
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
        word2vec_tgt = embeddingHandler.create_embedding(sentences_tgt, vocab_tgt)
        embeddingHandler.save_embedding(word2vec_tgt, tgt_embedding_output_path)
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
    print('Word2Vec dimension:', word2vec_dim)
    print('-------------------------------')

    ################## create dataset ######################
    batch_size = 64
    num_epochs = 12
    print('Creating dataset...')
    print('Number of training examples: ', training_size)

    # create training set for encoder (source)
    sentences_src_as_ids = embeddingHandler.convert_sentences_to_ids(dic_src, sentences_src)
    train_set_src = create_dataset(sentences_src_as_ids)

    # create training set for decoder (target)
    sentences_tgt_as_ids = embeddingHandler.convert_sentences_to_ids(dic_tgt, sentences_tgt)
    # for sentence_as_ids in sentences_tgt_as_ids:  # add </s> id to the end of each sentence of target language
    #     sentence_as_ids.append(eos_vocab_id)
    train_set_tgt = create_dataset(sentences_tgt_as_ids)
    # padding matrix
    target_weights = create_dataset([np.ones(len(sentence) + 1) for sentence in sentences_tgt_as_ids])

    # create dataset contains both previous training sets
    train_dataset = tf.data.Dataset.zip((train_set_src, train_set_tgt, target_weights))
    train_dataset = train_dataset.shuffle(buffer_size=training_size, seed=1)
    # train_dataset = train_dataset.shuffle(buffer_size=training_size)
    train_dataset = train_dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(batch_size, ([None], [None], [None])))
    train_iter = train_dataset.make_initializable_iterator()
    x_batch, y_batch, padding_matrix = train_iter.get_next()
    print('-------------------------------')

    #################### build graph ##########################
    hidden_size = 128  # number of hidden unit
    print('Building graph...')

    first_layer_encoder = BidirectionalEncoder(embedding_src,
        fw_cell=LSTMcell(hidden_size=hidden_size, input_size=word2vec_dim, batch_size=batch_size),
        bw_cell=LSTMcell(hidden_size=hidden_size, input_size=word2vec_dim, batch_size=batch_size)
    )
    second_layer_encoder = EncoderBasic(
        LSTMcell(hidden_size=hidden_size, input_size=2*hidden_size, batch_size=batch_size),
        embedding_src
    )
    attention_decoder = GlobalAttentionDecoder(
        LSTMcell(hidden_size=hidden_size, input_size=word2vec_dim, batch_size=batch_size),
        vocab_size=len(vocab_tgt),
        embeddings=embedding_tgt
    )
    global_step = tf.Variable(0, trainable=False, name='global_step')
    first_layer_hid_states = first_layer_encoder.encode(x_batch)
    second_layer_hid_states = second_layer_encoder.extract_more_info(first_layer_hid_states)
    logits, _, labels = attention_decoder.decode(y_batch, second_layer_hid_states, second_layer_hid_states[-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # shape=[time, batch]
    apply_penalties = tf.transpose(cross_entropy) * tf.cast(padding_matrix, tf.float32)
    loss = tf.reduce_sum(apply_penalties) / batch_size
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)  # derivation of loss by paramsz
    max_gradient_norm = 5
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    starting_rate = 1.0
    decay_epochs = 4  # decay learning rate on every n epochs exclude first n epochs
    decay_step = (training_size // batch_size) * decay_epochs  # num_step_in_single_epoch * n
    learning_rate = tf.train.exponential_decay(learning_rate=starting_rate, global_step=global_step,
                                               decay_steps=decay_step, decay_rate=0.8, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    #################### train ########################
    log_frequency = 100
    model_path = "./checkpoint/model"
    checkpoint_path = "./checkpoint"
    loss_epochs = tf.TensorArray(tf.float32, size=num_epochs, dynamic_size=True)
    training_epoch = tf.Variable(0, trainable=False, name='training_epoch')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
            print('...............Restored from checkpoint')
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
                    # print('Step {0}: loss={1} lr={2}'.format(step, l, lr))
                    if step % log_frequency == 0:
                        print('Step {0}: loss={1} lr={2}'.format(step, l, lr))
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


train_model()
