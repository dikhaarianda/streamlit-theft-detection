from keras import backend as K

class EvaluationMetrics:
  @staticmethod
  def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  @staticmethod
  def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

  @staticmethod
  def f1_score(y_true, y_pred):
    precision = EvaluationMetrics.precision(y_true, y_pred)
    recall = EvaluationMetrics.recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))