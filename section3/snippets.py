import tensorflow as tf

# ===============================================
# Previously was snippets.py of: 3_2_RNNs
# ===============================================

# i = input_gate, j = new_input, f = forget_gate, o = output_gate
# Get 4 copies of feeding [inputs, m_prev] through the "Sigma" diagram.
# Note that each copy has its own distinct set of weights.
lstm_matrix = self._linear1([inputs, m_prev])
i, j, f, o = tf.split(
    value=lstm_matrix, num_or_size_splits=4, axis=1)
# Feed each of the gates through a sigmoid.
i = sigmoid(i)
f = sigmoid(f + self._forget_bias)
o = sigmoid(o)

c = f * c_prev + i * self._activation(j)
m = o * self._activation(c)

new_state = LSTMStateTuple(c, m)
return m, new_state

# ===============================================
# RNN illustration
# ===============================================

hidden_size = 32


def rnn_step(x, h_prev):
    # Project inputs to each have dimension hidden_size.
    combined_inputs = tf.layers.Dense(hidden_size)(tf.concat([x, h_prev], axis=1))
    # Compute the next hidden state.
    h = tf.tanh(combined_inputs)
    return h


# ===============================================
# Bidirectional RNNs
# ===============================================
outputs_tuple, final_state_tuple = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=tf.nn.rnn_cell.LSTMCell(128),
    cell_bw=tf.nn.rnn_cell.LSTMCell(128),
    inputs=inputs,
    dtype=tf.float32)
# Concatenate the forward and backward outputs.
# Shape: (batch_size, max_seq_len, 2 * state_size)
outputs = tf.concat(outputs_tuple, -1)

# ===============================================
# Stacked RNNs
# ===============================================


def lstm_cell():
    return tf.nn.rnn_cell.LSTMCell(128)


cell = tf.nn.rnn_cell.MultiRNNCell([
    lstm_cell() for _ in range(2)])
outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

