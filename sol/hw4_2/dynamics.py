import tensorflow as tf
import numpy as np

EPS = np.finfo(np.float32).eps


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    # Predefined function to build a feedforward neural network
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.sess = sess
        self.normalization = normalization
        self.batch_size    = batch_size
        self.iterations    = iterations
        self.learning_rate = learning_rate

        # assuming continuous obs and actions
        obs_dim = env.observation_space.shape[0]
        # obs_dim = 1 if len(obs_dim) == 0 else obs_dim[0]
        ac_dim = env.action_space.shape[0]
        # ac_dim = 1 if len(ac_dim) == 0 else ac_dim[0]
        self.obs_t_ph   = tf.placeholder(tf.float32, shape=[None, obs_dim])
        self.act_t_ph   = tf.placeholder(tf.float32, shape=[None, ac_dim])
        self.input_t_ph = tf.concat([self.obs_t_ph, self.act_t_ph], axis=1)
        self.delta_t_ph = tf.placeholder(tf.float32, shape=[None, obs_dim])
        self.network    = build_mlp(self.input_t_ph, obs_dim, 'dynamics', n_layers,
                                    size, activation, output_activation)

        self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.network - self.delta_t_ph), axis=0))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states,
        (unnormalized)actions, (unnormalized)next_states and fit the
        dynamics model going from normalized states, normalized actions
        to normalized state differences (s_t+1 - s_t)
        """
        # self._normalization takes the following form
        # mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action
        obs          = np.concatenate([item['observations'] for item in data])
        nobs         = np.concatenate([item['next_observations'] for item in data])
        actions      = np.concatenate([item['actions'] for item in data])

        deltas       = nobs - obs

        norm_states  = (obs     - self.normalization[0])/(self.normalization[1] + EPS)
        norm_deltas  = (deltas  - self.normalization[2])/(self.normalization[3] + EPS)
        norm_actions = (actions - self.normalization[4])/(self.normalization[5] + EPS)

        for i in range(self.iterations):
            dataset       = tf.data.Dataset.from_tensor_slices((norm_states, norm_actions, norm_deltas))
            batch_dataset = dataset.shuffle(norm_states.shape[0]).batch(self.batch_size)
            iterator      = batch_dataset.make_one_shot_iterator()
            next_element  = iterator.get_next()
            loss = 0
            idx  = 0
            while True:
                try:
                    obs, action, delta = self.sess.run(next_element)
                except tf.errors.OutOfRangeError:
                    break
                feed_dict = {self.obs_t_ph:   obs,
                             self.act_t_ph:   action,
                             self.delta_t_ph: delta}

                loss += self.sess.run(self.loss, feed_dict)
                idx  += 1
                self.sess.run(self.optimizer, feed_dict)
            print('    Dyn fit -- avg loss {} :: epoch {}'.format(loss / idx, i))

    def predict(self, states, actions):
        """
        Write a function to take in a batch of (unnormalized) states
        and (unnormalized) actions and return the (unnormalized) next
        states as predicted by using the model
        """
        norm_states  = (states  - self.normalization[0])/(self.normalization[1] + EPS)
        norm_actions = (actions - self.normalization[4])/(self.normalization[5] + EPS)

        feed_dict = {self.obs_t_ph: norm_states, self.act_t_ph: norm_actions}

        norm_delta = self.sess.run(self.network, feed_dict)
        next_states = (norm_delta * self.normalization[3]) + self.normalization[2] + states
        return next_states
