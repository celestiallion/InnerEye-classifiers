import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv('Log_V1.log')

epochs = df['epoch']
train_acc = df['predictions_categorical_accuracy']
val_acc = df['val_predictions_categorical_accuracy']

plt.plot(epochs, train_acc, linewidth=1.5, color='steelblue', label='Training accuracy')
plt.plot(epochs, val_acc, linewidth=1.5, color='orangered', label='Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
