import numpy as np
import matplotlib.pyplot as plt


path='./results/logger.txt'
data=np.genfromtxt(path,delimiter=',')
acc=data[:,1]
val_acc=data[:,3]
loss=data[:,2]
val_loss=data[:,4]



plt.figure('Loss Diagram',dpi=600)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss,color='blue',marker='.',label='Train Data')
plt.plot(val_loss,color='red',marker='^',label='Validation Data')
plt.grid()
plt.legend()
plt.savefig('./results/loss_diagram.jpg')


plt.figure('Accuracy Diagram',dpi=600)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(acc,color='blue',marker='.',label='Train Data')
plt.plot(val_acc,color='red',marker='^',label='Validation Data')
plt.grid()
plt.legend()
plt.savefig('./results/accuracy_diagram.jpg')




