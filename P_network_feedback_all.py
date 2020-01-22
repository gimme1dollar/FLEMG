from _modules import Sensor, Encoder, Processor, Analysis
import numpy as np
from datetime import datetime
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, './data/encode_8ch_202001091942.csv')
now = datetime.now()

#Feature variable setting
subject = "A"
emg_dim = 8
flex_dim = 5
learning_rate = 0.01
iteration = 2000
seq_length = 3
input_dim = (emg_dim*2+flex_dim)*seq_length
stack_dim = 2
hidden_dim = 100
output_dim = emg_dim+flex_dim
#
Encoder = Encoder.encoder(emg_dim = emg_dim ,flex_dim = flex_dim)
Preprocessor = Processor.preprocessor()
Network = Processor.network_feedback_all()

Preprocessor.load(filename)
Preprocessor.scale()
trainIndex, trainData, trainLabel = Preprocessor.preprocess_feedback_all()
trainLength = int(len(trainIndex) * 0.7)
testIndex, testData, testLabel = trainIndex[trainLength:], trainData[trainLength:], trainLabel[trainLength:]
trainIndex, trainData, trainLabel = trainIndex[:trainLength], trainData[:trainLength], trainLabel[:trainLength]

print(f"Index example: {trainIndex[0].reshape(-1)}\n"
      f"Data example: {trainData[0]} \n"
      f"Label example: {trainLabel[0]}\n")

with Network.graph.as_default():
    Network.construct_placeholders(learning_rate=learning_rate,hidden_dim=hidden_dim,stack_dim=stack_dim)
    print("Model Constructed\n")
    #sav_loc = './model/' + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    model_name = now.strftime("%Y%m%d_%H%M")    
    model_dir = './model/' + Network.__class__.__name__ 
    try:
    # Create target Directory
        os.mkdir(model_dir)
        print("Directory " , model_dir ,  " Created\n") 
        model_dir = './model/' + Network.__class__.__name__ + '/' + subject
        try:
            os.mkdir(model_dir)
            print("Directory" , model_dir , "Created\n")
        except FileExistsError:
            print("Directory " , model_dir ,  " already exists\n")

    except FileExistsError:
        model_dir = './model/' + Network.__class__.__name__ + '/' + subject
        try:
            os.mkdir(model_dir)
            print("Directory" , model_dir , "Created\n")
        except FileExistsError:
            print("Directory " , model_dir ,  " already exists\n")
    model_dir = model_dir + '/' +model_name
    loss=Network.train_network(trainData, trainLabel,iteration, model_dir)

    print("Train Over\n")
    print("Model " + model_name + " saved\n")

    prediction, rmse_val = Network.infer(testData, testLabel)
    #p = Analysis.plotter(np.asarray(prediction), testLabel, testIndex, Network.output_dim)
    p = Analysis.plotter(Network.__class__.__name__,learning_rate=learning_rate,iteration=iteration, seq_length=seq_length , stack_dim=stack_dim, hidden_dim=hidden_dim, rmse=rmse_val,prediction=np.asarray(prediction),label=testLabel, index=testIndex , flex_dim=flex_dim)

    
    fig_dir = "./result/"+Network.__class__.__name__
    try:
    # Create target Directory
        os.mkdir(fig_dir)
        print("Directory " , fig_dir ,  " Created\n") 
        fig_dir = "./result/"+Network.__class__.__name__+"/"+ subject
        try:
            os.mkdir(fig_dir)
            print("Directory",fig_dir,"Created\n")
        except FileExistsError:
            print("Directory",fig_dir,"already exists\n")

    except FileExistsError:
        fig_dir = "./result/"+Network.__class__.__name__+"/"+ subject
        try:
            os.mkdir(fig_dir)
            print("Directory",fig_dir,"Created\n")
        except FileExistsError:
            print("Directory",fig_dir,"already exists\n")
    train_fig_name = now.strftime("train%Y%m%d_%H%M")
    train_fig_dir = fig_dir + '/' + train_fig_name    
    p.plot_training_graph(loss,iteration,figloc=train_fig_dir)
    print("Figure "+train_fig_name+" saved\n")

    test_fig_name = now.strftime("test%Y%m%d_%H%M")
    test_fig_dir = fig_dir + '/' + test_fig_name
    p.plot_comparison(subplot_row = 2, size = (20,10), figloc=test_fig_dir)
    print("Figure "+test_fig_name+" saved")
    test_data_name = now.strftime("test%Y%m%d_%H%M.csv")
    test_data_dir = fig_dir + '/' + test_data_name
    test_data=np.concatenate((testIndex[:,0],testLabel,np.asarray(prediction)),axis=1)
    np.savetxt(test_data_dir,test_data,delimiter=',')
    print(f"Data {test_data_name} saved")