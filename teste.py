from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_reais =  [1,0,1,0,1,1,0,0,1,0]
y_preditos=[1,0,0,0,1,1,1,0,1,1]
mc= confusion_matrix(y_reais,y_preditos)
plt.figure(figsize=(4,2))
sns.heatmap(mc,annot=True,cmap='Blues',fmt='d',cbar=False)
plt.title('Matriz da Confusão')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.show()