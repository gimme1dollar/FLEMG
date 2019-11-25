from modules import Sensor, Encoder, Processor, Analysis
import numpy as np
from datetime import datetime

Preprocessor = Processor.preprocessor()
Network = Processor.network()

Preprocessor.load('./data/full.csv')
testIndex, testData, testLabel = Preprocessor.preprocess()
print(f"Index example: {trainIndex[0].reshape(-1)}\n"
      f"Data example: {trainData[0]} \n"
      f"Label example: {trainLabel[0]}\n")

with Network.graph.as_default():
    Network.construct_placeholders()
    print("Model Constructed\n")

    Network.restore('./model/1124(2)')
    prediction, rmse_val = Network.infer(testData, testLabel)
    p = Analysis.plotter(np.asarray(prediction), testLabel, testIndex, 6)

    now = datetime.now()
    figname = "./result/prediction(" + str(int(rmse_val)) + ")_" + str(now.year) + str(now.month) + str(now.day) + '-' + str(now.hour) + str(now.minute)
    p.plot_comparison(subplot_row = 2, size = (20,10), figloc=figname)