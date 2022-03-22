# drop-coalescence-surrogate-model-and-LA

Codes are written by Yilin Zhuang and Sibo Cheng, including snippets from several other authors.

In this work, reduced-order modelling techniques, namely proper orthogonal decomposition and convolutional neural networks, are applied to compress the recorded images into low-dimensional spaces. Recurrent neural networks are then employed to build a surrogate model of drop interactions by learning the dynamics of compressed variables in the reduced-order space. To incorporate real-time observations, we developed an ensemble-based Latent Assimilation algorithm scheme. With the help of ensemble-based data assimilation techniques, the novel approach improve the prediction results by adjusting the starting point of the next time-level forecast.

The trainned CAE is included in the CAE/, the encoded test data is included in data/test_data/test_0-7.npy. LSTM_MSE.h5 is the LSTM model, and the training data are named as "video_5_train_Explabel_set.npy", "video_5_train_Expsample_set.npy".
Please refer to the LSTM_model.ipynb for the detailed procedure.
