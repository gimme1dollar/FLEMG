from _modules import Sensor, Encoder, Processor, Analysis
import numpy as np
from datetime import datetime
now = datetime.now()

Preprocessor = Processor.preprocessor()
Network = Processor.network()

Preprocessor.load('./data/full.csv')
trainIndex, trainData, trainLabel = Preprocessor.preprocess()
testIndex, testData, testLabel = trainIndex[2000:], trainData[2000:], trainLabel[2000:]
trainIndex, trainData, trainLabel = trainIndex[:2000], trainData[:2000], trainLabel[:2000]

print(f"Index example: {trainIndex[0].reshape(-1)}\n"
      f"Data example: {trainData[0]} \n"
      f"Label example: {trainLabel[0]}\n")

# Cross-Validation
target_rmse = 50
rmse_val = 1000

with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    e = 0
    while (rmse_val >= target_rmse):
        train_size = int(len(data) * 0.8)
        start = (e * int(len(data) * 0.1)) % len(data)
        print("[epoch: {}]   start: {} out of data_size: {}".format(e, start, len(data)))

        if start + train_size < len(data):
            train_set = data[start: start + train_size]
            validation_set = np.asarray(np.concatenate((data[start + train_size:], data[: start]), axis=0).tolist())
        else:
            train_set = np.asarray(
                np.concatenate((data[start:], data[: train_size + start - len(data)]), axis=0).tolist())
            validation_set = data[train_size + start - len(data): start]

        trainT, trainX, trainY = build_dataset_default(train_set, seq_length)
        validationT, validationX, validationY = build_dataset_default(validation_set, seq_length)

        # Training step
        for idx in range(len(trainX)):
            _, step_loss = sess.run([train, loss], feed_dict={X: [trainX[idx]], Y: [trainY[idx]]})
            if idx % 5000 == 0:
                print("step: {} loss: {}".format(idx, step_loss))

        # train_predict = sess.run(Y_pred, feed_dict={X: trainX})

        prediction = []
        for idx in range(len(validationX)):
            test_predict = sess.run(Y_pred, feed_dict={X: [validationX[idx]]})
            prediction.append(test_predict[0])

        # RMSE
        rmse_val = sess.run(rmse, feed_dict={targets: validationY, predictions: prediction})
        print("### RMSE: {}\n".format(rmse_val))
        e += 1

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, path.join(notebooks_base_dir, "/checkpoints/Cross_Validation.ckpt"))
    print('Trained Model Saved.')