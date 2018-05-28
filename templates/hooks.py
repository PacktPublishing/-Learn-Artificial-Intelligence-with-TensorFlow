import sys
import tensorflow as tf


class TutorialRunHook(tf.train.SessionRunHook):
    """A SessionRunHook made for tutorial purposes (showcase how it all works).

    Sources used when making this class:
        1) https://stackoverflow.com/questions/45532365)
        2) https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/python/training/session_run_hook.py#L70

    Pseudocode showing execution order of methods:

    call hooks.begin()
    sess = tf.Session()
    call hooks.after_create_session()
    while not stop is requested:  # py code: while not mon_sess.should_stop():
        call hooks.before_run()
        try:
            results = sess.run(merged_fetches, feed_dict=merged_feeds)
        except (errors.OutOfRangeError, StopIteration):
            break
        call hooks.after_run()
    call hooks.end()
    sess.close()
    """

    def __init__(self):
        pass

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be
        launched in the session. The hook can modify the graph by
        adding new operations to it. After the begin() call the graph
        will be finalized and the other callbacks can not modify the
        graph anymore.

        Second call of begin() on the same graph, should not change the graph.
        """
        pass

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
        pass

    def before_run(self, run_context):
        """Called before each call to session.run().

        You can return from this call a SessionRunArgs object indicating
        ops or tensors to add to the upcoming run() call. These ops/tensors
        will be run together with the ops/tensors originally passed to the
        original run() call. The run args you return can also contain feeds
        to be added to the run() call. Note that at this point graph is finalized and
        you can not add ops.

        Args:
            run_context: A SessionRunContext object, with properties
                - original_args: tf.train.SessionRunArgs containing the arguments
                    about to be passed to session.run(...), which we can add to.
                - session: the tf.Session object being used.
                - stop_requested: bool

        Returns:
            None or a SessionRunArgs object. SessionRunArgs is a namedtuple with
                fields: fetches, feed_dict, options, corresponding to the arguments
                of a sess.run(...) call.
        """
        pass

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The run_values argument contains results of requested ops/tensors by before_run().

        The run_context argument is the same one send to before_run call.
        run_context.request_stop() can be called to stop the iteration.

        If session.run() raises any exceptions then after_run() is not called.

        Args:
            run_context: A SessionRunContext object, with properties
                - original_args: tf.train.SessionRunArgs.
                - session: tf.Session.
                - stop_requested: bool.
            run_values: A SessionRunValues object, with properties
                - results: return values from previous session.run(...) call.
                - options: RunOptions passed to the previous session.run(...) call.
                - run_metadata: RunMetadata passed to the previous session.run(...) call.
        """
        pass

    def end(self, session):
        """
        Called at the end of session.

        The session argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If session.run() raises exception other than OutOfRangeError or
        StopIteration then end() is not called.

        Note the difference between end() and after_run() behavior when
        session.run() raises OutOfRangeError or StopIteration.
        In that case end() is called but after_run() is not called.

        Args:
            session: A TensorFlow Session that will be soon closed.
        """
        pass

