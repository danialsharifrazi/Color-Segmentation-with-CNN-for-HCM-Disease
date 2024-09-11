def NetPlot(net_history):
    import numpy as np
    import matplotlib.pyplot as plt

    history=net_history.history
    losses=history['loss']
    val_losses=history['val_loss']
    accuracies=history['accuracy']
    val_accuracies=history['val_accuracy']

    plt.figure('Loss Diagram')
    plt.title('Loss of Deep Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['Train Data','Validation Data'])

    plt.figure('Accuracy Diagram')
    plt.title('Accuracy of Deep Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['Train Data','Validation Data'])       
    plt.show()


