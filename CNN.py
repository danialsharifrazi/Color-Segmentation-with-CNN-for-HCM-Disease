def DeepCNN(x_data,y_data):
    import datetime
    from sklearn.metrics import auc, classification_report,confusion_matrix, roc_curve
    from sklearn.model_selection import train_test_split
    from keras.callbacks import CSVLogger


    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=42)
    print(f'train: {x_train.shape}  {y_train.shape}')
    print(f'test: {x_test.shape}  {y_test.shape}')

    lst_loss=[]
    lst_acc=[]
    lst_reports=[]
    lst_AUC=[]
    lst_matrix=[]
    lst_times=[]
    from keras.utils import np_utils

    y_train=np_utils.to_categorical(y_train)
    y_test=np_utils.to_categorical(y_test)

    calback=CSVLogger(f'./results/logger.log')


    from keras.layers import (Activation, BatchNormalization, Conv1D, Conv2D,
                            Dense, Dropout, Flatten, Input, MaxPool1D, MaxPool2D)
    from keras.layers.advanced_activations import LeakyReLU
    from keras.models import Model, Sequential
    from keras.regularizers import l1, l2


    model=Sequential()
    model.add(Conv1D(32,3,padding='same',activation='relu',strides=2,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Dropout(0.25))
    model.add(Conv1D(64,3,padding='same',activation='relu',strides=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128,3,padding='same',activation='relu',strides=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
    model.add(Dropout(0.25))
    model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
    model.add(Dropout(0.25))
    model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='sigmoid'))


    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            
    start=datetime.datetime.now()
    net_history=model.fit(x_train, y_train, batch_size=512, epochs=50,validation_split=0.2,callbacks=[calback])
    end=datetime.datetime.now()

    import PlotHistory

    PlotHistory.NetPlot(net_history)
    model.save(f'./results/CNN.h5')

    test_loss, test_acc=model.evaluate(x_test,y_test)
    lst_acc.append(test_acc)
    lst_loss.append(test_loss)

    predicts=model.predict(x_test)
    predicts=predicts.argmax(axis=1)
    actuals=y_test.argmax(axis=1)

    fpr,tpr,_=roc_curve(actuals,predicts)
    a=auc(fpr,tpr)
    lst_AUC.append(a)

    r=classification_report(actuals,predicts)
    lst_reports.append(r)

    c=confusion_matrix(actuals,predicts)
    lst_matrix.append(c)

    training_time=end-start
    lst_times.append(training_time)

    path=f'./results/CNN_Results.txt' 
    f1=open(path,'a')
    f1.write('\nAccuracies: '+str(lst_acc)+'\nLosses: '+str(lst_loss)+'\nAUCs: '+str(lst_AUC)+'\n')
    f1.write('\n\nMetrics for all Folds: \n\n')
    for i in range(len(lst_reports)):
        f1.write(str(lst_reports[i]))
        f1.write('\n\nTraining Time: '+str(lst_times[i]))
        f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n___________________________\n')
    f1.close()



