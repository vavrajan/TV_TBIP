# Import global packages
import os
import time

import pandas as pd
from absl import app
from absl import flags

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
import matplotlib.pyplot as plt

flags.DEFINE_integer("reest_seed", default=314159, help="Random seed to be used.")
flags.DEFINE_string("data_name", default='simulation', help="Name of the dataset.")
flags.DEFINE_string("data", default='with_quoting', help="Name of the dataset to be used.")
FLAGS = flags.FLAGS

### Define the scenarios for ideological positions
def get_ideal(scenario, author_party, ideal, s):
    match scenario:
        case "zero":
            return np.zeros(author_party.shape)
        case "party":
            return 1.0*(author_party == 'D') + 0.0*(author_party == 'I') - 1.0*(author_party == 'R')
        case "diverge":
            if s <= 104:
                return np.zeros(author_party.shape)
            else:
                return 0.2*(s-104) * (author_party == 'D') + 0.0 * (author_party == 'I') - 0.2*(s-104) * (author_party == 'R')
        case "estimate":
            return ideal

def main(argv):
    del argv
    print(tf.__version__)

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.data_name, FLAGS.data)

    # scenarios = ["zero", "party", "diverge", "estimate"]
    # scenarios = ["estimate"]
    scenarios = "party"
    seed = FLAGS.reest_seed

    # Betas from the last session used for all sessions
    neutral_topics = np.load(os.path.join(source_dir, '114', 'input', "neutral_topic_mean.npy"))
    beta = neutral_topics
    positive_topics = np.load(os.path.join(source_dir, '114', 'input', "positive_topic_mean.npy"))
    negative_topics = np.load(os.path.join(source_dir, '114', 'input', "negative_topic_mean.npy"))
    eta = 10 * 0.5 * (positive_topics - negative_topics) # todo try multiplying with 10

    for s in range(97, 115):
        print('Starting sampling session ' + str(s) + '.')
        ## Directory setup
        s_dir = os.path.join(source_dir, str(s))
        input_dir = os.path.join(s_dir, 'input')
        output_dir = os.path.join(s_dir, 'output')
        counts_dir = os.path.join(s_dir, 'simulated_counts')
        if not os.path.exists(s_dir):
            os.mkdir(s_dir)
        if not os.path.exists(input_dir):
            os.mkdir(input_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(counts_dir):
            os.mkdir(counts_dir)

        ## Load data
        # inputs
        author_indices = np.load(os.path.join(input_dir, "author_indices.npy")).astype(np.int32)
        author_data = np.loadtxt(os.path.join(input_dir, "author_map.txt"),
                                 dtype=str, delimiter=" ", usecols=[0, 1, -1])
        author_party = np.char.replace(author_data[:, 2], '(', '')
        author_party = np.char.replace(author_party, ')', '')
        author_map = np.char.add(author_data[:, 0], author_data[:, 1])

        # model parameters
        theta = tf.cast(tf.constant(pd.read_csv(os.path.join(input_dir, "thetas.csv"), index_col=0).to_numpy()), "float32")
        # neutral_topics = np.load(os.path.join(input_dir, "neutral_topic_mean.npy"))
        # beta = neutral_topics
        # positive_topics = np.load(os.path.join(input_dir, "positive_topic_mean.npy"))
        # negative_topics = np.load(os.path.join(input_dir, "negative_topic_mean.npy"))
        estimated_ideal = np.load(os.path.join(input_dir, "ideal_point_mean.npy"))
        # eta = 0.5 * (positive_topics - negative_topics)
        ## Let's try sampled etas
        # eta_dist = tfp.distributions.normal(location=0, scale=eta.scale)

        ## Trigger different scenarios
        for scenario in scenarios:
            print('Scenario = ' + scenario)
            # Create idealogical positions depending on scenario and session number s
            ideal = tf.cast(tf.constant(get_ideal(scenario, author_party, estimated_ideal, s)), "float32")
            # Get Poisson rates and sum them over topics
            rate = tf.math.reduce_sum(tf.math.exp(
                theta[:, :, tf.newaxis] + beta[tf.newaxis, :, :] +
                eta[tf.newaxis, :, :] * tf.gather(ideal, author_indices)[:, tf.newaxis, tf.newaxis]
            ), axis=1)
            # Create the Poisson distribution with given rates
            count_distribution = tfp.distributions.Poisson(rate=rate)
            # Sample the counts
            seed, sample_seed = tfp.random.split_seed(seed)
            counts = count_distribution.sample(seed=sample_seed)
            print(counts.shape)
            sparse_counts = sparse.csr_matrix(counts)
            print(sparse_counts.shape)
            sparse.save_npz(os.path.join(counts_dir, "counts_" + scenario + ".npz"), sparse_counts)

        print('Session ' + str(s) + ' finished')


if __name__ == '__main__':
    app.run(main)




