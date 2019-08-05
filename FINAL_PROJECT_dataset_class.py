import numpy as np
import os
import sys
import matplotlib.pyplot as plt
def load_embeddings(embed_file):
    # store all the pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    with open(os.path.join('glove.6B.%sd.txt' % 50),'r') as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
        print('Found %s word vectors.' % len(word2vec))
    return word2vec

class Dataset():
    def __init__(self,
                data_file_train = 'augmented_training_birds.npy',
                data_file_test='augmented_test_birds.npy',
		text_file_test = 'augmented_test_birds_text.npy',
		text_file_train = 'augmented_training_birds_text.npy',
                load_texts=True,
                names_file = 'names_final.npy',
                embed_file = 'embedding_matrix.npy',
                train_test_split = 0.2,
                batch_size = 64,
                transpose = False,
                aug_factor = 5,
                load_embeddings = False):
        self.batch_size = batch_size
        self.names = np.load(names_file)
        self.n_samples = self.names.shape[0]*aug_factor
        print("loading %s samples"%self.n_samples)
        #self.test_set_size = int(self.n_samples*0.2)
        self.test_set_size = 100*aug_factor
        shuffle_data = np.random.permutation(self.names.shape[0]*aug_factor)
        self.load_texts= load_texts
        if transpose:
            self.test_images = np.transpose(np.load(data_file_test),(0,3,1,2))/255.0
            print("test images shape: ",self.test_images.shape)
            self.train_images = np.transpose(np.load(data_file_train),(0,3,1,2))/255.0
            print("train images shape: ",self.train_images.shape)

        else:
            self.test_images = np.load(data_file_test)/255.0
            print("test images shape: ",self.test_images.shape)
            self.train_images = np.load(data_file_train)/255.0
            print("train images shape: ",self.train_images.shape)
        print("samples successfully loaded")

        if load_texts:
            print('loading texts')
            self.test_texts = np.load(text_file_test)
            print("loaded test text data: ",self.test_texts.shape)
            self.train_texts = np.load(text_file_train)
            print("loaded train text data: ",self.train_texts.shape)

            if load_embeddings:
                self.embeddings = np.load(embed_file)

    def shuffle(self):
        self.shuffle_idx = np.random.permutation(range(self.n_samples-self.test_set_size))

    def load_data(self, train=True):

        if train == True:
            self.shuffle()

            if self.load_texts:
                return zip(np.array_split(self.train_images[self.shuffle_idx],
                              ((self.n_samples-self.test_set_size)//self.batch_size)),
                       np.array_split(self.train_texts[self.shuffle_idx],
                              ((self.n_samples-self.test_set_size)//self.batch_size)))
            else:
                return np.array_split(self.train_images[self.shuffle_idx],
                              ((self.n_samples-self.test_set_size)//self.batch_size))
        else:
            if self.load_texts:

                return zip(np.array_split(self.test_images,
                              (self.test_set_size//self.batch_size)),
                       np.array_split(self.test_texts,
                              (self.test_set_size//self.batch_size)))
            else:
                return np.array_split(self.test_images,
                              (self.test_set_size//self.batch_size))
