from tensorflow.keras.utils import Sequence
import numpy as np

dim_img = 512

class TrainGenerator(Sequence):

    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(dim_img,dim_img), batch_size = 1, n_channels = 4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = range(list_IDs)
        self.list_IDs = range(list_IDs)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(Batch_ids)

        return X, Y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        VOLUME_SLICES = 1
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 1))

        path = f'./data/train/'
        
        # Generate data
        for c, i in enumerate(Batch_ids):
        
            for j in range(VOLUME_SLICES):
                 X[j + VOLUME_SLICES*c, :, :, 0] = np.load(f'{path}abdos/{i}_abdo.npy')
                 X[j + VOLUME_SLICES*c, :, :, 1] = np.load(f'{path}portals/{i}_portal.npy')
                 X[j + VOLUME_SLICES*c, :, :, 2] = np.load(f'{path}arteriels/{i}_arteriel.npy')
                 X[j + VOLUME_SLICES*c, :, :, 3] = np.load(f'{path}veineux/{i}_veineux.npy')

                 Y[j + VOLUME_SLICES*c, :, :, 0] = np.load(f'{path}segmentations/{i}_segmentation.npy')

        return X, Y
