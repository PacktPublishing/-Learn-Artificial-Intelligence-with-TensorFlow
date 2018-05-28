# Section 3: Natural Language Processing and Recurrent Neural Networks

Brief file descriptions:
- `3_1_embeddings.py`: simple autoencoder that we implemented to illustrate how to work with embeddings in TensorFlow.
- `3_4_classification_part_one.py`: simple text classification model that uses `tf.nn.dynamic_rnn`.
- `3_5_classification_part_two.py`: more advanced text classification model that uses `tf.nn.bidirectional_dynamic_rnn`, `MultiRNNCell`, and `DropoutWrapper`.


## Additional Resources

### 3.1: Word Embeddings

- [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec) TensorFlow tutorial.
- [Embeddings](https://www.tensorflow.org/programmers_guide/embedding) Programmer's Guide.
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/). Main site that includes links to pretrained word vectors and the paper by J. Pennington, R. Socher, and C. Manning (2014).
- [tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) documentation.
- [Section 5.2 (GloVe) of my notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf), based partially on Lecture 3 of Stanford's CS224n, and on the original GloVe paper.


### 3.2: Recurrent Neural Networks

- [Recurrent Neural Networks](https://www.tensorflow.org/tutorials/recurrent) TensorFlow tutorial.
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Chris Olah.
- [Chapter 10 of Deep Learning](http://www.deeplearningbook.org/contents/rnn.html): Sequence Modeling: Recurrent and Recursive Nets.
- [Section 2.5 (RNNs) of my notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf), where I explicitly work through the derivations for the gradients from Chapter 10 of the Deep Learning book.

### 3.3: Bidirectionality and Stacking RNNs

- [bidirectional_dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn) documentation.
- [MultiRNNCell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell) documentation.

### 3.4/3.5: Models for Text Classification

- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) homepage.
- [example.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto) source code.
- [RNN and Cells](https://www.tensorflow.org/api_guides/python/contrib.rnn) TensorFlow module.
