#comparing actual response values(y_test) with predicted response values(y_pred)
from sklearn import metrics 
y_test = [1,0,0,1,1,1,1]
y_predicted = [0,0,0,0,1,1,0]
conf_matrix = metrics.confusion_matrix(y_test,y_predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=[False,True])
print("Confusion Matrix:",conf_matrix)

import matplotlib.pyplot as plt
cm_display.plot()
# plt.show()

Accuracy = metrics.accuracy_score(y_test,y_predicted)
Precision = metrics.precision_score(y_test,y_predicted,average = 'micro')
Sensitivity_recall = metrics.recall_score(y_test,y_predicted,average = 'micro')
Specificity = metrics.recall_score(y_test,y_predicted,average = 'micro')
F1_score = metrics.f1_score(y_test,y_predicted,average = 'micro')
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_predicted,pos_label = 1) #pos_label stands for positive label
AUC = metrics.auc(fpr,tpr)
MCC = metrics.matthews_corrcoef(y_test,y_predicted) #Matthews correlation coefficient (MCC)

#metrics
print("Accuracy:",Accuracy,"\nPrecision:",Precision,"\nSensitivity_recall:",Sensitivity_recall,"\nSpecificity:",Specificity,"\nF1_score:",F1_score,"\nAUC:",AUC,"\nMCC:",MCC)

