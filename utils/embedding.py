import numpy as np
from gensim.models import Word2Vec


class Embedding:
    def load_vocab(self, vocab_file):
        """
        Load vocabulary from file
        :param vocab_file: path to predefined vocabulary file
        :return:
        :vocab : list of words in vocabulary
        :dic : dictionary which maps words to indices corresponding to vocabulary, i.e 'home': index_in_vocab
        """
        vocab = []
        dic = {}
        with open(vocab_file, encoding="utf8") as file:
            i = 0
            for line in file:
                vocab.append(line[:-1])  # remove \n symbol and add to list
                dic[vocab[-1]] = i  # assign newly added word to dictionary
                i += 1
        return vocab, dic

    def load_sentences(self, train_file):
        """
        Load sentences from file and convert to list of sentences with each element is list of word
        :param train_file: path to file
        :return: array 2-D
        """
        sentences = []
        with open(train_file, encoding='utf8') as file:
            for line in file:
                line = line.split()
                sentences.append(line)
        return sentences

    def create_embedding(self, sentences, vocab, vector_size=200, window=10):
        """
        Create word embedding and save to local machine
        :param sentences: list of sentence
        :param vocab: list of word
        :param vector_size: size of vector representation
        :param window: sliding window used for train
        :return: Word2Vec object
        """
        vocab = list(map(lambda x: [x], vocab))
        # train model
        model = Word2Vec(size=vector_size, window=window, min_count=1, sg=1, hs=0)
        model.build_vocab(vocab)
        model.train(sentences, total_examples=len(sentences), epochs=20)
        return model

    def save_embedding(self, model, output_file):
        model.save(output_file)

    def load_embedding(self, path):
        """
        Load word embedding from file
        :param path: absolute file path
        :return:
        :obj: `~gensim.models.word2vec.Word2Vec`
                Returns the loaded model as an instance of :class: `~gensim.models.word2vec.Word2Vec`.
        """
        return Word2Vec.load(path)

    def parse_embedding_to_list_from_vocab(self, word2vec, vocab):
        """
        Parse Word2Vec object into list of embedding
        :param word2vec: Word2Vec object
        :param vocab: list of word
        :return: embedding list correspond to vocab
        """
        embeddings = []
        for w in vocab:
            embeddings.append(word2vec[w])
        embeddings = np.asarray(embeddings)
        return embeddings

    def find_vector_word(self, word, embedding):
        """
        find a vector represent for word
        :param word: string
        :param embedding: FastText object
        :return:a np.array
        """
        words = embedding.wv.vocab
        if word in words:
            return embedding[word]
        else:
            return embedding["<unk>"]

    def convert_sentences_to_ids(self, dic, list_sentences):
        """
        Convert sentences into indices which corresponding to vocabulary
        :param dic: dictionary which maps words->ids corresponding to vocabulary
        :param list_sentences: list contains sentences separated by new line
        :return:
        :sentences : list of sentence converted to indices
        """
        sentences_as_ids=[]
        for sentence in list_sentences:
            sentence_ids = []
            for word in sentence:
                try:
                    sentence_ids.append(dic[word])
                except KeyError:
                    sentence_ids.append(1)  # 1 is index of <unk> in vocabulary
            sentences_as_ids.append(sentence_ids)
        return sentences_as_ids

    def ids_to_words(self, list_ids, vocab):
        return [vocab[id] for id in list_ids]

    def words_to_ids(self, list_words, dic):
        return [dic[word] for word in list_words]
