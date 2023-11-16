from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data_metrics.csv')
df.head()

# add columns predicted_RF and predicted_LR
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()


def find_tp(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_fn(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_fp(y_true, y_pred):
    # counts the number of true positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_tn(y_true, y_pred):
    # counts the number of true positives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


# check results
print('TP:', find_tp(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_fn(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_fp(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_tn(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    tp = find_tp(y_true, y_pred)
    fn = find_fn(y_true, y_pred)
    fp = find_fp(y_true, y_pred)
    tn = find_tn(y_true, y_pred)
    return tp, fn, fp, tn


def struk_confusion_matrix(y_true, y_pred):
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])


# check results
cm = struk_confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print('Confusion Matrix:\n', cm)

assert np.array_equal(struk_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_RF.values)), 'struk_confusion_matrix() is not correct for RF'
assert np.array_equal(struk_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_LR.values)), 'struk_confusion_matrix() is not correct for LR'


# accuracy_score
def struk_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


# check results
assert struk_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values,
                                                                                              df.predicted_RF.values), 'struk_accuracy_score failed on RF'
assert struk_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values,
                                                                                              df.predicted_LR.values), 'struk_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (struk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (struk_accuracy_score(df.actual_label.values, df.predicted_LR.values)))


# recall_score

def struk_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall


# check results
assert struk_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values,
                                                                                          df.predicted_RF.values), 'struk_recall_score failed on RF'
assert struk_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values,
                                                                                          df.predicted_LR.values), 'struk_recall_score failed on LR'
print('Recall RF: %.3f' % (struk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (struk_recall_score(df.actual_label.values, df.predicted_LR.values)))


# precision_score
def struk_precision_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    tp, fn, fp, tn = find_conf_matrix_values(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision


# check results
assert struk_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values,
                                                                                                df.predicted_RF.values), 'struk_recall_score failed on RF'
assert struk_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values,
                                                                                                df.predicted_LR.values), 'struk_recall_score failed on LR'
print('Precision RF: %.3f' % (struk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (struk_precision_score(df.actual_label.values, df.predicted_LR.values)))


# f1_score
def struk_f1_score(y_true, y_pred):
    # Calculate precision and recall
    precision = struk_precision_score(y_true, y_pred)
    recall = struk_recall_score(y_true, y_pred)

    # Calculate F1 score using the formula: 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score


# Check results
assert struk_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values,
                                                                                  df.predicted_RF.values), 'struk_f1_score failed on RF'
assert struk_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values,
                                                                                  df.predicted_LR.values), 'struk_f1_score failed on LR'

print('F1 RF: %.3f' % (struk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (struk_f1_score(df.actual_label.values, df.predicted_LR.values)))

# threshold change
print('scores with threshold = 0.5')
print('Accuracy RF:%.3f' % (struk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (struk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF:%.3f' % (struk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (struk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF:%.3f' % (struk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (struk_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF:%.3f' % (struk_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (struk_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

# conclusions

# roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# use of area metric under curve
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

#graphics
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()