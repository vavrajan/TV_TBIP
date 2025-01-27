import functools
import time
import os
import pandas as pd
import csv

# import tensorflow_version 1.15
import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import non_negative_factorization
import tensorflow as tf
import tensorflow_probability as tfp

from absl import app
from absl import flags

import sys
# current path is './simulation/03_sess_97.py'
# Add ./simulation to the list of paths so that 'tbip.py' can be found
sys.path.append(os.path.dirname(sys.path[0]))
from tbip import *

## FLAGS
flags.DEFINE_enum("scenario", default="_zero", enum_values=["", "_zero", "_party", "_diverge", "_estimate"],
                  help="Simulation scenario, one of:"
                       "= use the original counts,"
                       "_zero = all ideological positions will be zero,"
                       "_party = all ideological positions will be determined by party -1, 0, 1,"
                       "_diverge = ideal will start at zero and from session 10* grows away from zero,"
                       "_estimate = use estimates of ideal from the corresponding sessions.")
flags.DEFINE_integer("reest_seed", default=0, help="Random seed to be used.")
flags.DEFINE_string("data_name", default='simulation', help="Name of the dataset.")
flags.DEFINE_string("version", default='with_quoting', help="Name of the different version of the dataset to be used.")
flags.DEFINE_integer("session", default=98, help="Session number (time-point) where to start.")
flags.DEFINE_float("eps", default=1, help="Epsilon (precision) of the Adam algorithm.")
flags.DEFINE_float("learningrate", default=0.0001, help="Learning rate of the Adam algorithm.")
flags.DEFINE_integer("numtopics", default=25, help="Number of topics.")
flags.DEFINE_integer("batchsize", default=512, help="Number of documents in a single batch.")
flags.DEFINE_integer("maxsteps", default=250000, help="Maximum number of iterations.")
flags.DEFINE_integer("printsteps", default=25000, help="How often to print the steps.")
FLAGS = flags.FLAGS

def main(argv):
    del argv
    print(tf.__version__)

    tf.test.gpu_device_name()
    tf.set_random_seed(FLAGS.reest_seed)
    random_state = np.random.RandomState(0)

    num_topics = FLAGS.numtopics
    # parameters for ADAM
    eps = FLAGS.eps
    learningrate = FLAGS.learningrate
    batch_size = FLAGS.batchsize
    max_steps = FLAGS.maxsteps
    print_steps = FLAGS.printsteps

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.data_name, FLAGS.version)
    #path to tbip.py, to change as needed
    # py_file = os.path.join(project_dir, 'jae_revision')
    py_file = os.path.join(project_dir, 'python', FLAGS.data_name)

    sys.path.append(os.path.abspath(py_file))
    import tbip

    scenario = FLAGS.scenario
    s = FLAGS.session
    #for s in range(98, 115):
    old_dir = os.path.join(source_dir, str(s - 1), 'output')
    s_dir = os.path.join(source_dir, str(s))
    input_dir = os.path.join(s_dir, 'input')
    output_dir = os.path.join(s_dir, 'output')
    counts_dir = os.path.join(s_dir, 'simulated_counts')


    tf.reset_default_graph()  # to be used if graph building doesn't renew
    tf.set_random_seed(FLAGS.reest_seed)  # after tf_reset_graph
    random_state = np.random.RandomState(FLAGS.reest_seed)  # after tf_reset_graph

    objective_topic_loc = np.load(os.path.join(old_dir, "objective_topic_loc" + scenario + ".npy"))
    beta = np.exp(objective_topic_loc)

    author_data = np.loadtxt(os.path.join(input_dir, "author_map.txt"),
                             dtype=str, delimiter=" ", usecols=[0, 1, -1])
    author_party = np.char.replace(author_data[:, 2], '(', '')
    author_party = np.char.replace(author_party, ')', '')

    counts_path = os.path.join(counts_dir, 'counts' + scenario + '.npz')
    X = sparse.load_npz(counts_path)
    # X = X.astype('float64')
    W, H, n_iter = non_negative_factorization(X, n_components=num_topics,
                                              init='custom',
                                              random_state=0,
                                              update_H=False,
                                              H=beta.astype(
                                                  'float32'))  # paul:we had astype('double') but returned error

    W = W.astype('float32')
    W = W + 1e-3  # causes 0 value otherwise
    initial_document_loc = W  # theta
    objective_topic_loc_prev = beta.astype('float32')  # don't need, use objective_topic_loc

    ideological_topic_loc_prev = np.load(os.path.join(old_dir, 'ideological_topic_loc' + scenario + '.npy'))  # eta
    ideological_topic_loc_prev = tf.convert_to_tensor(ideological_topic_loc_prev)

    # using tbip.build_input_pipeline
    # original source code available at: https://github.com/keyonvafa/tbip
    (iterator, author_weights, vocabulary, author_map,
     num_documents, num_words, num_authors) = tbip.build_input_pipeline(
        counts_path,
        input_dir,
        batch_size,
        random_state,
        counts_transformation='nothing')
    document_indices, counts, author_indices = iterator.get_next()


    # Create Lognormal variational family for document intensities (theta).
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))  # theta from NMF
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, num_topics],
        initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)
    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)

    # Create Lognormal variational family for objective topics (beta).
    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(np.log(objective_topic_loc_prev)),  # from session 97
        trainable=True)
    objective_topic_scale_logit = tf.get_variable(
        "objective_topic_scale_logit",
        shape=[num_topics, num_words],
        initializer=tf.initializers.random_normal(mean=-2, stddev=1.),
        dtype=tf.float32,
        trainable=True)
    objective_topic_scale = tf.nn.softplus(objective_topic_scale_logit)
    objective_topic_distribution = tfp.distributions.LogNormal(
        loc=objective_topic_loc,
        scale=objective_topic_scale)

    # Create Gaussian variational family for ideological topics (eta).
    ideological_topic_loc = tf.get_variable(
        "ideological_topic_loc",
        initializer=ideological_topic_loc_prev,
        # initialized only ideological_topic_loc so that var type doesn't change
        trainable=True)
    ideological_topic_scale_logit = tf.get_variable(
        "ideological_topic_scale_logit",
        shape=[num_topics, num_words],
        dtype=tf.float32,
        trainable=True)
    ideological_topic_scale = tf.nn.softplus(ideological_topic_scale_logit)
    ideological_topic_distribution = tfp.distributions.Normal(
        loc=ideological_topic_loc,
        scale=ideological_topic_scale)

    # Create Gaussian variational family for ideal points (x).
    ideal_point_loc = tf.get_variable(
        "ideal_point_loc",
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale_logit = tf.get_variable(
        "ideal_point_scale_logit",
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale = tf.nn.softplus(ideal_point_scale_logit)
    ideal_point_distribution = tfp.distributions.Normal(
        loc=ideal_point_loc,
        scale=ideal_point_scale)

    # Approximate ELBO.
    elbo = tbip.get_elbo(counts,
                         document_indices,
                         author_indices,
                         author_weights,
                         document_distribution,
                         objective_topic_distribution,
                         ideological_topic_distribution,
                         ideal_point_distribution,
                         num_documents,
                         batch_size)
    loss = -elbo

    optim = tf.train.AdamOptimizer(learning_rate=learningrate, epsilon=eps)

    # original_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-04)
    # optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
    train_op = optim.minimize(loss)

    document_mean = document_loc + document_scale ** 2 / 2

    neutral_mean = objective_topic_loc + objective_topic_scale ** 2 / 2

    positive_mean = (objective_topic_loc +
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 +
                      ideological_topic_scale ** 2) / 2)  # intialized

    negative_mean = (objective_topic_loc -
                     ideological_topic_loc +
                     (objective_topic_scale ** 2 +
                      ideological_topic_scale ** 2) / 2)  # intialized

    print('Learning for session ' + str(s) + ' and scenario ' + scenario)

    loss_vals = []
    ideal_df = pd.DataFrame(data={"speaker": author_map, "party": author_party})

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    start_time = time.time()
    for step in range(max_steps):
        (_, elbo_val) = sess.run([train_op, elbo])
        loss_vals.append(elbo_val)
        duration = (time.time() - start_time) / (step + 1)
        if step % print_steps == 0 or step == max_steps - 1:
            print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec/step)".format(
                step, elbo_val, duration))
        if (step + 1) % print_steps == 0 or step == max_steps - 1:
            (document_topic_mean, neutral_topic_mean, negative_topic_mean, positive_topic_mean,
             ideal_point_mean, ots, otl, itl, its) = sess.run([document_mean, neutral_mean, negative_mean,
                                                               positive_mean, ideal_point_loc, objective_topic_scale,
                                                               objective_topic_loc,
                                                               ideological_topic_loc,
                                                               ideological_topic_scale])
            ideal_df['ideal' + str(step)] = ideal_point_mean
            ideal_df.to_csv(os.path.join(output_dir, 'ideal_step_by_step' + scenario + '.csv'), sep=',', index=False)
            np.save(os.path.join(output_dir, "document_topic_mean" + scenario + ".npy"), document_topic_mean)
            np.save(os.path.join(output_dir, "neutral_topic_mean" + scenario + ".npy"), neutral_topic_mean)
            np.save(os.path.join(output_dir, "negative_topic_mean" + scenario + ".npy"), negative_topic_mean)
            np.save(os.path.join(output_dir, "positive_topic_mean" + scenario + ".npy"), positive_topic_mean)
            np.save(os.path.join(output_dir, "ideal_point_mean" + scenario + ".npy"), ideal_point_mean)
            np.save(os.path.join(output_dir, "objective_topic_scale" + scenario + ".npy"), ots)
            np.save(os.path.join(output_dir, "objective_topic_loc" + scenario + ".npy"), otl)
            np.save(os.path.join(output_dir, "ideological_topic_loc" + scenario + ".npy"), itl)
            np.save(os.path.join(output_dir, "ideological_topic_scale" + scenario + ".npy"), its)
            loss_df = pd.DataFrame(data={"loss": loss_vals})
            loss_df.to_csv(os.path.join(output_dir, 'loss_values' + scenario + '.csv'), sep=',',
                           index=False)  # save loss values as a csv file
    print("session " + str(s) + " done!")


if __name__ == '__main__':
    app.run(main)

