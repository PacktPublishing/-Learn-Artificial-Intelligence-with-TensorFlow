import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from pprint import pprint

OPTIMIZER_CLS_NAMES = {
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
    "Ftrl": tf.train.FtrlOptimizer,
    "Momentum": tf.train.MomentumOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
}


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))


def create_gradient_summaries(gradients):
    for gradient, variable in gradients:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        if grad_values is not None:
            var_name = variable.name.replace(':', '_')
            tf.summary.scalar('gradient_norm/%s' % var_name,
                               tf.global_norm([grad_values]))


def create_adam_summaries(opt, learning_rate):
    for slot in ['m', 'v']:
        t_list = [opt.get_slot(v, slot) for v in tf.trainable_variables()
                  if v is not None]
        slot_norm = tf.global_norm(t_list, name='{}_norm_op'.format(slot))
        tf.summary.scalar('{}_norm'.format(slot), slot_norm)

        if slot == 'v':
            effective_lr = tf.divide(learning_rate, 1e-8 + tf.sqrt(slot_norm))
            tf.summary.scalar('effective_lr', effective_lr)


def optimize_loss(loss, learning_rate, optimizer, clip_gradients=None):
    """Simplified version of tf.contrib.layers.optimize_loss, for
    illustration purposes.

    Args:
        loss: (float) initial value for the learning rate.
        optimizer: (str) one of the allowed optimizers in OPTIMIZER_CLS_NAMES.
        clip_gradients: (float) if given, clip gradients such that their norm
            is at most `clip_gradients` for any given variable.

    Returns:
        train_op: the training operation that computes gradients and updates weights.
    """

    global_step = tf.train.get_global_step()
    with tf.variable_scope('OptimizeLoss', values=[loss, global_step]):
        update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        loss = control_flow_ops.with_dependencies(list(update_ops), loss)

        lr = tf.get_variable(
            'learning_rate', (), trainable=False,
            initializer=tf.constant_initializer(learning_rate))

        opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
        # `gradients` is list of (gradient, variable) pairs, where
        # `gradient` is the gradient for `variable`.
        gradients = opt.compute_gradients(
            loss, var_list=tf.trainable_variables())

        if clip_gradients is not None:
            gradients = _clip_gradients_by_norm(gradients, clip_gradients)
            tf.summary.scalar('global_norm/clipped_gradient_norm',
                              tf.global_norm(list(zip(*gradients))[0]))

        # Generate a scalar summary for each variable, giving its gradient norm.
        create_gradient_summaries(gradients)

        # Create gradient updates.
        grad_updates = opt.apply_gradients(
            gradients, global_step=global_step, name='train')

        if optimizer == 'Adam':
            create_adam_summaries(opt, learning_rate)

        # Ensure the train_op computes grad_updates.
        train_op = control_flow_ops.with_dependencies([grad_updates], loss)
        return train_op
