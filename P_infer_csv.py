from modules import Sensor, Encoder, Processor, Analysis
import numpy as np
from datetime import datetime
now = datetime.now()

Encoder = Encoder.encoder(flex_dim = 5)
Preprocessor = Processor.preprocessor()
Network = Processor.network()

Preprocessor.load('./data/full(5).csv')
trainIndex, trainData, trainLabel = Preprocessor.preprocess()
trainLength = int(len(trainIndex) * 0.7)
testIndex, testData, testLabel = trainIndex[trainLength:], trainData[trainLength:], trainLabel[trainLength:]
trainIndex, trainData, trainLabel = trainIndex[:trainLength], trainData[:trainLength], trainLabel[:trainLength]

print(f"Index example: {trainIndex[0].reshape(-1)}\n"
      f"Data example: {trainData[0]} \n"
      f"Label example: {trainLabel[0]}\n")

with Network.graph.as_default():
    Network.construct_placeholders(learning_rate=0.15)
    print("Model Constructed\n")
    sav_loc = './model/' + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    Network.train_network(trainData, trainLabel, 2, sav_loc)
    print("Train Over\n")

    prediction, rmse_val = Network.infer(testData, testLabel)
    p = Analysis.plotter(np.asarray(prediction), testLabel, testIndex, 5)
    figname = "./result/prediction(" + str(int(rmse_val)) + ")_" + str(now.year) + str(now.month) + str(now.day) + '_' + str(now.hour) + str(now.minute)
    p.plot_comparison(subplot_row = 2, size = (20,10), figloc=figname)