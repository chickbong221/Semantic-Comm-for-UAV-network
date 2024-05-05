import random
import scipy
import scipy.io
import numpy as np
import Environment_marl
from Environment_marl import Environ
import os
from replay_memory import ReplayMemory
import sys
from collections import deque

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, BatchNormalization, Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.metrics import mse
from keras.callbacks import LearningRateScheduler
from keras import backend as K

np.random.seed(110803)
random.seed(22010)

# Cấu hình tùy chọn bộ nhớ GPU
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

REPLAY_MEMORY_SIZE = 70_000
MIN_REPLAY_MEMORY_SIZE = 8_000
MINIBATCH_SIZE = 2000

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 0.99
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

# ################## SETTINGS ######################

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'marl_model'

env = Environ()

n_UAVs = 4
n_RB = 2
n_power_level = len(env.power_dB_list)
n_priority_level = len(env.priority_level_list)

env.new_random_game()  # initialize parameters in env

n_episode = 540
n_step_per_episode = 200
epsi_final = 0.02
epsi_anneal_length = int(400)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

######################################################

def get_state(env, idx=0):
    """ Get state from the environment """

    Q_graph, n_images = env.get_ori_Qgraph(env.UAV_image_select[idx, :]) 
    Q_graph = Q_graph / 10_000
    n_images = n_images / 10
    Q_data = env.Q_data[idx] / 1_000
    UAV_interference = (env.UAV_Interference_all[idx, :] + 75) / 100    
    UAV_h_i = env.h_i 
    distance = env.distance[:, idx] / 1_000
    Gamma_i = env.Gamma_i[:, idx] * 10

    # print(Q_graph, "Q_graph")
    # print(n_images)
    # print(Q_data, "Q_data")
    # print(UAV_interference, "UAV_interference")
    # print(UAV_h_i,"UAV_h_i")
    # print(distance)
    # print(Gamma_i)
    
    return np.concatenate((np.array([Q_data]), np.array(distance), np.array(Gamma_i), np.array([Q_graph]), np.array([n_images]), UAV_interference.flatten(), np.reshape(np.asarray([UAV_h_i]), -1), UAV_interference.flatten()))
        

# ----------------------------------------------------------
n_hidden_1 = 512
n_hidden_2 = 258
n_hidden_3 = 126
n_input = len(get_state(env=env))
n_output = n_RB * n_power_level * n_priority_level

# ============== Training network ========================
def create_model(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
    model = Sequential([
        Dense(units= n_hidden_1, input_shape=(n_input, ), activation='relu'),
        BatchNormalization(),
        Dense(units= n_hidden_2, activation='relu'),
        BatchNormalization(),
        Dense(units= n_hidden_3, activation='relu'),
        BatchNormalization(),
        Dense(units= n_output)
    ])
    model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

def predict(sess, s_t, ep, test_ep = False):

    state = np.array(s_t).reshape(-1, *s_t.shape)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB * n_power_level * n_priority_level)
    else:
        pred_actions = sess.predict(state)[0]
        pred_action = np.argmax(pred_actions)
    return pred_action

def q_learning_mini_batch(current_agent, current_sess, current_sess_p):
    """ Training a sampled mini-batch """
    
    if len(current_agent.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return
    
    minibatch = random.sample(current_agent.replay_memory, MINIBATCH_SIZE)
    
    if current_agent.double_q:  # double q-learning

        # current_states = np.array([transition[0] for transition in minibatch])
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list_pre = current_sess.predict(current_states) 
        
        # future_states = np.array([transition[1] for transition in minibatch])
        future_states = np.array([transition[1] for transition in minibatch])
        future_qs_list_pre = current_sess.predict(future_states)
        future_qs_list_pre_max_index = np.argmax(future_qs_list_pre, axis=1)

        future_qs_list_tar = current_sess_p.predict(future_states) 

        X = []
        Y = []

        for index, (batch_s_t, batch_s_t_plus_1, batch_action, batch_reward) in enumerate(minibatch):

            new_q = batch_reward + current_agent.discount * future_qs_list_tar[index, future_qs_list_pre_max_index[index]]
            # print(future_qs_list_pre_max_index[index])

            current_qs_pre = current_qs_list_pre[index]
            current_qs_pre[batch_action] = new_q

            X.append(batch_s_t)
            Y.append(current_qs_pre)
    # else: #DQN
    #     current_states = np.array([transition[0] for transition in minibatch])
    #     current_qs_list_pre = current_sess.predict(current_states) 
        
    #     future_states = np.array([transition[1] for transition in minibatch])
    #     future_qs_list_pre = current_sess.predict(future_states)
    #     future_qs_list_pre_max_index = np.argmax(future_qs_list_pre, axis=1)

    #     X = []
    #     Y = []

    #     for index, (batch_s_t, batch_s_t_plus_1, batch_action, batch_reward) in enumerate(minibatch):

    #         new_q = batch_reward + current_agent.discount * future_qs_list_pre[index, future_qs_list_pre_max_index[index]]

    #         current_qs_pre = current_qs_list_pre[index]
    #         current_qs_pre[batch_action] = new_q

    #         X.append(batch_s_t)
    #         Y.append(current_qs_pre)
    history = current_sess.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=1, shuffle=False)
    
    with open("lr.txt","a") as f:
            f.write(str(np.round(float(current_sess.optimizer.learning_rate), 5)) + '\n' )
    loss_values = history.history['loss']
    return loss_values

def update_target_q_network(sess, sess_p):
    """ Update target q network once in a while """
    
    sess_p.set_weights(sess.get_weights())

# def save_models(sess, model_path):
#     """ Save models to the current directory with the name filename """

#     current_dir = os.path.dirname(os.path.realpath(__file__))
#     model_path = os.path.join(current_dir, "model/" + model_path)
#     if not os.path.exists(os.path.dirname(model_path)):
#         os.makedirs(os.path.dirname(model_path))
#     sess.save(model_path)


# def load_models(sess, model_path):
#     """ Restore models from the current directory with the name filename """

#     dir_ = os.path.dirname(os.path.realpath(__file__))
#     model_path = os.path.join(dir_, "model/" + model_path)
#     sess = load_model(model_path)

# --------------------------------------------------------------
agents = []
sesses = []
sesses_p = []

for ind_agent in range(n_UAVs):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    # print(len(get_state(env)))
    agents.append(agent)
    
    sess = create_model(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    sesses.append(sess)
    sess_p = create_model(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    sesses_p.append(sess_p)

# for i in range(n_UAVs):
#     print(sesses[i].summary())

# ------------------------- Training -----------------------------
record_reward = np.zeros([n_episode*n_step_per_episode, 1])

if IS_TRAIN:
    if os.path.exists("lr.txt"):
        os.remove("lr.txt")
    if os.path.exists("c_per_ep_2RB.txt"):
        os.remove("c_per_ep_2RB.txt")
    if os.path.exists("Reward_per_ep_2RB.txt"):
        os.remove("Reward_per_ep_2RB.txt")
    if os.path.exists("Gamma_per_ep_2RB.txt"):
        os.remove("Gamma_per_ep_2RB.txt")
    if os.path.exists("relay_per_ep_2RB.txt"):
        os.remove("relay_per_ep_2RB.txt")
    for i_episode in range(n_episode):
        if i_episode == 150:
            new_learning_rate = 0.0003
            for sess in sesses:
                K.set_value(sess.optimizer.lr, new_learning_rate)
        elif i_episode == 300:
            new_learning_rate = 0.00005
            for sess in sesses:
                K.set_value(sess.optimizer.lr, new_learning_rate)
        average_all_relay = 0
        average_each_relay = np.zeros((n_UAVs))
        gamma_i = np.zeros((n_UAVs))
        Reward_per_ep = 0
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < np.ceil(MIN_REPLAY_MEMORY_SIZE / n_step_per_episode):
            epsi = 1
        elif (i_episode - np.ceil(MIN_REPLAY_MEMORY_SIZE / n_step_per_episode)) < epsi_anneal_length:
            epsi = 1 - (i_episode - np.ceil(MIN_REPLAY_MEMORY_SIZE / n_step_per_episode)) * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final

        for i_step in range(n_step_per_episode):
            time_step = i_episode*n_step_per_episode + i_step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([n_UAVs, 3], dtype='int8')
            for i in range(n_UAVs):
                state = get_state(env, i)
                state_old_all.append(state)
                action = predict(sesses[i], state, epsi)
                action_all.append(action)

                action_all_training[i, 0] = action % (n_RB)  # chosen RB
                a = np.floor(action/(n_power_level*n_RB))
                action_analyze_for_power = action - a*(n_power_level*n_RB)
                action_all_training[i, 1] = int(np.floor(action_analyze_for_power / n_RB)) # power level               
                action_all_training[i, 2] = int(np.floor(action / (n_RB * n_power_level))) # priority level
                gamma_i[i] += env.priority_level_list[action_all_training[i, 2]]

            # print(state_old_all)
            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward, b, d = env.act_for_training(action_temp)
            record_reward[time_step] = train_reward

            Reward_per_ep += train_reward
            average_all_relay += b
            for i in range(n_UAVs):
                average_each_relay[i] += d[i]

            env.renew_user_distance()
            env.renew_user2UAV_channel()
            env.Compute_Q_data()
            env.Qgraph_gen()    
            env.renew_BS_channel()
            env.Compute_Interference(action_temp)

            for i in range(n_UAVs):
                state_old = state_old_all[i]
                action = action_all[i]
                state_new = get_state(env, i)
                agents[i].update_replay_memory((state_old, state_new, action, train_reward))

                # training this agent
                if time_step % mini_batch_step == mini_batch_step-1:
                    loss_val_batch = q_learning_mini_batch(agents[i], sesses[i], sesses_p[i])

                if time_step % target_update_step == target_update_step-1:
                    update_target_q_network(sesses[i], sesses_p[i])
                    if i == 0:
                        print('Update target Q network...')
                            
        with open("Reward_per_ep_2RB.txt","a") as f:
            f.write(str(i_episode) + ', ' + str(Reward_per_ep) +'\n' )

        with open("Gamma_per_ep_2RB.txt","a") as f:
            f.write(str(i_episode) + ', ' + str(gamma_i[0]/n_step_per_episode) + ', ' + str(gamma_i[1]/n_step_per_episode) + ', ' + str(gamma_i[2]/n_step_per_episode) + ', ' + str(gamma_i[3]/n_step_per_episode) +'\n' )

        with open("relay_per_ep_2RB.txt","a") as f:
            f.write(str(i_episode) + ', ' + str(average_each_relay[0]/n_step_per_episode) + ', ' + str(average_each_relay[1]/n_step_per_episode) + ', ' + str(average_each_relay[2]/n_step_per_episode) + ', ' + str(average_each_relay[3]/n_step_per_episode) +'\n' )

        with open("c_per_ep_2RB.txt","a") as f:
            f.write(str(i_episode) + ', ' + str(average_all_relay/n_step_per_episode) +'\n' )    
    print('Training Done. Saving models...')





