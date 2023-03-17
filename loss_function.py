from math import gamma
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

beta = 0.25
alpha = 0.25
epsilon = 1e-5
smooth = 1

class SemanticLossFunction(object):

    def __init__(self):
        print ("Semantic loss functions initialized")

    def dice_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def dice_coef(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        return (2 * intersection) / (K.sum(y_true) + K.sum(y_pred))

    def jacard_coef_loss(self, y_true, y_pred):
        return 1 - self.jacard_coef(y_true, y_pred)

    def jacard_coef(self, y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        score = (intersection) / ( (K.sum(y_true) + K.sum(y_pred) - intersection))
        return score
      
    def sensitivity(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / ( (possible_positives + K.epsilon()))

    def specificity(self, y_true, y_pred):
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0 , 1)))
        return true_negatives / ( (possible_negatives + K.epsilon()))
