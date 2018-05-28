import os
import tensorflow as tf
from collections import deque


class MaxWindow:
    """Helps determine whether a value has increased by at least some amount
    more than the smallest value in a specified window size.
    Note that comparisons are not done against the oldest value in the
    window, since that would be sensitive to anomalous spikes/dips.
    """

    def __init__(self, window_size, epsilon):
        self.patience = window_size
        self.epsilon = epsilon
        self.window = deque(maxlen=self.patience)
        self.smallest_val = -float('inf')
        self._has_increased = True

    def update(self, current_val):
        # Only update smallest_val once we've collected `patience` entries.
        # Note that self.window will never exceed `self.patience` entries
        # by definition of collections.deque.
        if len(self.window) == self.patience:
            self.smallest_val = min(self.window)
        if current_val < self.smallest_val + self.epsilon:
            self._has_increased = False
        self.window.append(current_val)

    def has_increased(self):
        return self._has_increased


class EarlyStoppingHook(tf.train.SessionRunHook):
    """Custom SessionRunHook that will terminate training when accuracy
    is found above some threshold.
    N.B. Relies on existense of an 'acc_metric' collection in the default
    graph.
    """

    def __init__(self, metric, max_metric=0.99, patience=5, epsilon=5e-4):
        """
        Args:
            metric: tuple of (tensor, update_op) as returned by any
                tf.metrics function or any custom_ops metric.
            max_metric: (float) threshold for `metric` such that if
                `metric` exceeds `max_metric`, training will be terminated.
            patience: threshold number of runs to allow `metric` value to
                stay unchanged (in a row), plus or minus `epsilon`, before stopping.
            epsilon: small value used to determine whether `metric` has
                changed significantly.
        """
        self.metric = metric
        self.max_metric = max_metric
        self.max_window = MaxWindow(patience, epsilon)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.metric)

    def after_run(self, run_context, run_values):
        if not run_values.results:
            return

        metric_val, _ = run_values.results
        if metric_val > self.max_metric:
            msg = '{}={:.3f} is above threshold of {}.'.format(
                self.metric_name, metric_val, self.max_metric)
            self.stop(run_context, msg)

        self.max_window.update(metric_val)
        if not self.max_window.has_increased():
            msg = 'Metric {}={} < {}. '.format(
                self.metric_name, metric_val,
                self.max_window.smallest_val + self.max_window.epsilon)
            self.stop(run_context, msg)

    def stop(self, run_context, msg):
        tf.logging.info('EARLY STOPPING:\n\t{}\n'.format(msg))
        run_context.request_stop()
        raise SystemExit

    @property
    def metric_name(self):
        return self.metric[0].name


class StopEvalHook(tf.train.SessionRunHook):

    def __init__(self, should_stop, force_stop_op):
        self._should_stop = should_stop
        self._force_stop_op = force_stop_op

    def before_run(self, run_context):
        should_stop = run_context.session.run(self._should_stop)
        if should_stop:
            run_context.session.run(self._force_stop_op)
            run_context.request_stop()


class FreezingHook(tf.train.SessionRunHook):

    def __init__(self, output_node_name, output_path):
        self.output_node_name = output_node_name
        self.output_path = output_path
        print('WIGGA WIGGA WAT')
        self._do_it_yourself()

    def _do_it_yourself(self):
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer(),
                      tf.tables_initializer()])
            self.end(sess)

    def after_create_session(self, session, coord):
        """
        Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created.
        This has two essential differences with the situation in which begin is called:
            1. When this is called, the graph is finalized and ops can no
               longer be added to the graph.
            2. This method will also be called as a result of recovering a
               wrapped session, not only at the beginning of the overall session.

        Args:
            session: A TensorFlow Session that has been created.
            coord: A Coordinator object which keeps track of all threads.
        """
        print('WAT NOW SON')
        self.end(session)

    def end(self, session):
        print('Freezing model.')
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session, session.graph.as_graph_def(), [self.output_node_name])
        with tf.gfile.GFile(self.output_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        muh_x = session.graph.get_tensor_by_name('x:0')
        muh_preds = session.graph.get_tensor_by_name(self.output_node_name + ':0')
        tflite_model = tf.contrib.lite.toco_convert(
            output_graph_def, [muh_x],  [muh_preds])
        with tf.gfile.GFile(self.output_path.replace('.pb', '.tflite'), 'wb') as f:
            f.write(tflite_model)


class CustomOpsHook(tf.train.SessionRunHook):
    """Ultra simple hook that ensures certain ops get run at the correct
    time/place: exactly when the main graph ops are run in the same
    session.run call."""

    def __init__(self, ops):
        self._ops = ops

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self._ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS))


class ExportListener(tf.train.CheckpointSaverListener):
    """
    Example usage:
        ckpt_saver_hook = tf.train.CheckpointSaverHook(
            hparams.model_dir,
            save_steps=hparams.train_steps,
            listeners=[ExportListener(classifier)]
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.sir_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            features={'x': tf.placeholder(
                dtype=tf.int64,
                shape=[None, None],
                name='x')})

        self.exporter = tf.estimator.LatestExporter('exporter', self.sir_fn)

    def before_save(self, session, global_step_value):
        print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
        print('Done writing checkpoint.')
        self.exporter.export(
            estimator=self.estimator,
            export_path=os.path.join(self.estimator.model_dir, 'exports'),
            checkpoint_path=tf.train.latest_checkpoint(self.estimator.model_dir),
            eval_result=None, # NOT EVEN USED
            is_the_final_export=None) # NOT EVEN USED

