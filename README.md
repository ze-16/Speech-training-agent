# Speech-training-agent
This is a project to identify a mismatch between text sentiment and sentiment from audio features, to provide feedback based on the mismatch.

This project is not yet ready for inference and is still being developed.

# Tools used
```
--models: BERT and Wav2Vec2

--dataset: Goemotions used for text sentiment analysis

--epochs: number of epochs used is 3

--batch_size: batch size used is 64 

--schedular: used to slow down the learning rate during training

--cuda: whether or not to use GPU, default False

--max_len: max length set to 128

--class weights: the weigths used for each class is 1, 1, 2, 1.5

--learning rate: the rate used is 3e-5

--test size: test size used is 0.3

--activation function: the activation function used is ReLU

--dimensionality reduction: used to reduce the dimensions of the model from 768 to 32 before giving 4 outputs
```
The text only model is subject to the limitations of BERT base uncased, therefore relating back to project hypothesis (i.e acoustic features alongside text is needed to get a higher accuracy for emotion classification).

The variant of the audio model used (Wav2Vec2 base superb er) is already pre-trained for emotion classification, and it was initialised and the weights frozen.

The GPU used to speed up the training in the colab IDE is the A100 GPU.

