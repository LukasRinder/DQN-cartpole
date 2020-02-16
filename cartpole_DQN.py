import numpy as np
import tensorflow as tf
import datetime
import os
import matplotlib.pyplot as plt
from cartpole_model import CartPoleModel


class Backbone(tf.keras.Model):
    """
    Backbone of the Deep Q-Network (DQN) that approximates the Q-function.
    Takes 'num_states' inputs and outputs on Q-value for each action.
    """
    def __init__(self, num_states, hidden_units, num_actions):
        super(Backbone, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='RandomNormal'))  # TODO: ReLU layer ausprobieren

        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN(tf.Module):
    """
    Deep Q-Network.
    """
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, batch_size, lr):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.SGD(lr)
        self.gamma = gamma
        self.model = Backbone(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's_next': [], 'end': []}
        self.max_experiences = max_experiences

    def predict(self, inputs):
        """
        Get Q-values from backbone network.
        :param inputs: inputs for the backbone network, e.g. states.
        :return: outputs of the backbone network, e.g. num_action Q-values.
        """
        return self.model(tf.convert_to_tensor(inputs, tf.float32))

    def train(self, target_net):
        """
        Train with experience replay, e.g. replay using a randomized order removing correlation in observation sequence
        to deal with biased sampling
        :param target_net: target network.
        """
        experience_replay_enabled = True  # set False to disable experience replay
        if experience_replay_enabled:
            # sample random minibatch of transitions
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        else:
            n = len(self.experience['s'])
            if n < self.batch_size:
                ids = np.full(self.batch_size, n-1)
            else:
                ids = np.arange(max(0, n - self.batch_size), (n - 1), 1)

        states = tf.convert_to_tensor([self.experience['s'][i] for i in ids], tf.float32)
        actions = tf.convert_to_tensor([self.experience['a'][i] for i in ids], tf.float32)
        rewards = tf.convert_to_tensor([self.experience['r'][i] for i in ids], tf.float32)
        states_next = tf.convert_to_tensor([self.experience['s_next'][i] for i in ids], tf.float32)
        ends = tf.convert_to_tensor([self.experience['end'][i] for i in ids], tf.bool)

        # compute loss and perform gradient descent
        loss = self.gradient_update(target_net, states, actions, rewards, states_next, ends)

        return loss

    @tf.function
    def gradient_update(self, target_net, states, actions, rewards, states_next, ends):
        """
        Gradient update with @tf.function decorator for faster performance.
        """
        # make predictions with target network and get sample q for Q-function update, sample is different if epoch end
        target_network_enabled = True  # set False to disable target network
        if target_network_enabled:
            q_max = tf.math.reduce_max(target_net.predict(states_next), axis=1)
        else:
            q_max = tf.math.reduce_max(self.predict(states_next), axis=1)
        y = tf.where(ends, rewards, rewards + self.gamma * q_max)

        # perform gradient descent
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)

            # Q-values from training network for selected actions
            q_values = self.predict(states)
            selected_q_values = tf.math.reduce_sum(q_values * tf.one_hot(tf.cast(actions, tf.int32), self.num_actions), axis=1)

            loss = tf.math.reduce_sum(tf.square(y - selected_q_values))  # compute loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def get_action(self, states, epsilon):
        """
        Choose random action with probability 'epsilon', otherwise choose action with greedy policy, e.g. action that
        maximizes the Q-value function.
        :param states: observed states, e.g. [x, dx, th, dth].
        :param epsilon: probability of random action.
        :return: action
        """
        # take random action with probability 'epsilon'
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
            return action

        # else take action that maximizes the Q-function
        else:
            q_values = self.predict(np.atleast_2d(states))
            action = np.argmax(q_values)
            return action

    def add_experience(self, exp):
        """
        Add experience to experience history. If 'max_experiences' exceeded, remove first item and append current
        experience.
        :param exp: experience {'s': prev_observations, 'a': action, 'r': reward, 's_next': observations, 'end': end}.
        """
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)

        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, train_net):
        """
        Copy weights from train network to target network.
        :param train_net: model of train network.
        """
        variables_target = self.model.trainable_variables
        variables_train = train_net.model.trainable_variables

        for v_target, v_train in zip(variables_target, variables_train):
            v_target.assign(v_train.numpy())


def train_episode(cart_pole, train_net, target_net, epsilon, copy_steps, actions, iters_per_epoch):
    rewards = 0
    end = False
    state = cart_pole.get_state()
    losses = np.empty(iters_per_epoch)

    for n in range(iters_per_epoch):
        action = train_net.get_action(state, epsilon)  # select action random or after greedy policy
        force = actions[action]

        prev_state = state  # store old observations
        _, state, reward, abort = cart_pole.step(force)  # execute action, observe reward and next state
        rewards = rewards + reward

        if n == (iters_per_epoch-1) or abort:  # epoch ends
            end = True

        # store transitions
        exp = {'s': prev_state, 'a': action, 'r': reward, 's_next': state, 'end': end}
        train_net.add_experience(exp)
        losses[n] = train_net.train(target_net)

        # copy weights every 'copy_steps' to target network
        if n % copy_steps == 0:
            target_net.copy_weights(train_net)

        if abort:
            break

    mean_loss = np.mean(losses)

    return rewards, n, mean_loss


def test_policy(train_net, cart_pole, actions, time, video=False, printing=False, disturbance=False, f_disturbance=0.0, run_id=0):

    t_disturbance = 5.0

    N = int(time/cart_pole.dt_action)
    state = cart_pole.reset_env(std=0.0)

    rewards = 0
    s_traj = np.zeros((N, 4))
    a_traj = np.zeros(N)

    for i in range(N):
        t = round(i * cart_pole.dt_action)
        action = train_net.get_action(state, 0)  # epsilon = 0, e.g. no randomness
        force = actions[action]
        a_traj[i] = force
        s_traj[i, :] = state

        if int(t_disturbance) == int(t) and disturbance:
            t, state, reward, abort = cart_pole.step(force, f_disturbance)
        else:
            t, state, reward, abort = cart_pole.step(force, 0)
        rewards = rewards + reward

    if printing:
        print(f"Last state: {cart_pole.state}")
        print(f"Target range x: {cart_pole.x_target}")
        print(f"Target range theta: {cart_pole.th_target}")
        print(f"Testing steps: {i}, rewards {rewards}")
        print(f"Task accomplished: {cart_pole.task_accomplished}")
        print(f"Target range met: {cart_pole.target_range}")
        print(f"Steady state time: {cart_pole.steady_state_time}")
        print(f"Max. x overshoot: {cart_pole.x_overshoot}")
        print(f"Max. theta overshoot in °: {(180/np.pi) * cart_pole.th_overshoot}")

    if video:
        cart_pole.visualize(s_traj, name=str(run_id) + '_cart_pole_' + str(f_disturbance))  # create a video

    return rewards, cart_pole.task_accomplished, a_traj, s_traj


def main(checkpoint_dir=None, save=False, run_id=0):
    dt_action = 0.1
    dt_sim = 0.01
    t_abort = 30

    # create cart_pole environment
    cart_pole = CartPoleModel(dt_sim, dt_action, t_abort)
    num_states = len(cart_pole.reset_env())

    gamma = 0.9  # discount factor
    copy_steps = 25
    a1 = np.arange(-10.0, -2.0, 1.0)
    a2 = np.arange(-2.0, 2.0, 0.2)
    a3 = np.arange(2.0, 11.0, 1.0)
    actions = np.concatenate((a1, a2, a3))
    print(actions)

    num_actions = len(actions)
    print(num_actions)

    n_epochs = 1500
    iters_per_epoch = 100
    total_rewards = np.empty(n_epochs)
    epoch_iters = np.empty(n_epochs)

    hidden_units = np.array([256, 512, 512, 256])
    max_experiences = 50000  # replay memory capacity
    batch_size = 256

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, n_epochs, 1e-4, power=0.5)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'tensorboard_logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # initialize train (action-value function) and target network (target action-value function)
    train_net = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, batch_size, learning_rate_fn)
    target_net = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, batch_size, learning_rate_fn)
    target_net.copy_weights(train_net)  # initialize with same weights

    checkpoint_directory = 'checkpoints/' + current_time
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    opt = train_net.optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=train_net.model)

    epsilon = 0.2  # probability of selecting a random action

    epoch_task_accomplished = 0
    test_episodes = True  # set if the policy should be tested after each episode to plot the avg. accumulated reward
    plot_avg_reward = True  # plot if the avg. reward per epoch should be plotted
    time = 10  # time for video and testing the policy

    if isinstance(checkpoint_dir, str):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-1")
        checkpoint.restore(checkpoint_prefix)
        time = 10
        test_policy(train_net, cart_pole, actions, time, video=True, printing=True)
    else:
        for n in range(n_epochs):
            cart_pole.reset_env()  # initialize sequence

            total_reward, iterations, mean_loss = train_episode(cart_pole, train_net, target_net, epsilon, copy_steps,
                                                                actions, iters_per_epoch)
            epoch_iters[n] = iterations

            if test_episodes:
                total_reward, task_accomplished, _, _ = test_policy(train_net, cart_pole, actions, time)
                if task_accomplished and epoch_task_accomplished == 0:
                    epoch_task_accomplished = n
                    print(f"Task accomplished at epoch: {epoch_task_accomplished}")

            total_rewards[n] = total_reward
            avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()  # average reward of the last 100 episodes

            with summary_writer.as_default():
                tf.summary.scalar('episode reward', total_reward, step=n)
                tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)

            if n % 50 == 0:
                print(f"Episode: {n}, reward: {total_reward}, loss: {mean_loss}, iterations: {iterations}, eps: {epsilon}"
                      f", avg reward (last 100): {avg_rewards}")

        checkpoint.save(file_prefix=checkpoint_prefix)

        if plot_avg_reward:
            plt.figure()
            plt.plot(np.arange(n_epochs), total_rewards, linewidth=0.75)
            plt.xlabel("Training epochs")
            plt.ylabel("Accumulated reward per episode")
            plt.tight_layout()
            plt.savefig("plots/" + str(run_id) + "_" + current_time + "_AccumulatedReward.pdf")
            plt.close()

        if save:
            _, _, a_traj, s_traj = test_policy(train_net, cart_pole, actions, time, video=True, printing=True,
                                                     disturbance=False, run_id=run_id)
            _, _, a_traj_dist5, s_traj_dist5 = test_policy(train_net, cart_pole, actions, time, video=True, printing=True,
                                                     disturbance=True, f_disturbance=5.0, run_id=run_id)
            _, _, a_traj_dist10, s_traj_dist10 = test_policy(train_net, cart_pole, actions, time, video=True, printing=True,
                                                     disturbance=True, f_disturbance=10.0, run_id=run_id)


if __name__ == '__main__':
    checkpoint_dir = None  # set checkpoint directory to load from checkpoint
    save = True  # set save_dir if important training values should be saved
    n_trainings = 1

    # test gpu availability
    print(f"GPU available: {tf.test.is_gpu_available()}")

    for i in range(n_trainings):
        main(checkpoint_dir, save, i)
