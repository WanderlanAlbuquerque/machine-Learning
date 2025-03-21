import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Gerando valores reais e preditivos (1 = Positivo para Covid, 0 = Negativo para Covid)
y_true = np.random.choice([0, 1], size=20)  # Valores reais
y_pred = np.random.choice([0, 1], size=20)  # Valores preditos

# Criando a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Calculando métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Valor Predito')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão')
plt.show()

# Exibindo as métricas em um gráfico
metrics = ['Acurácia', 'Precisão', 'Sensibilidade (Recall)', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.ylabel('Valor')
plt.title('Métricas de Desempenho')
plt.show()

# Exibir os valores calculados
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Sensibilidade (Recall): {recall:.2f}")
print(f"F1 Score: {f1:.2f}")