import numpy as np
import time
import random
import math


np.random.seed(110803)

class Environ:

    def __init__(self):
        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.tau1 = 0.02
        self.tau2 = 0.08
        self.n_user = 20
        self.n_UAV = 4
        self.n_RB = 2
        self.Q_data = np.zeros((self.n_UAV))
        self.Q_graph = np.zeros((self.n_UAV))

        # User - UAV
        self.bandwidthdB_i_outside = 10**6
        self.power_user_dB = 15
        self.h_u = np.zeros((self.n_user,self.n_UAV))
        self.Gamma_list = np.asarray([-17,-18,-19])
        self.Gamma_i = np.zeros((self.n_user,self.n_UAV))

        # UAV - BS
        self.bandwidthdB_B_outside = 15 * 10**3
        self.SINR_UAV_BS = np.zeros((self.n_UAV))
        self.UAV_Interference_all = np.zeros((self.n_UAV, self.n_RB))
        self.h_i = np.zeros((self.n_UAV, self.n_RB))
        self.p_i = np.zeros((self.n_UAV))
        self.power_dB_list = [23, 15, 5, -100]
        self.priority_level_list = [0.1, 0.3, 0.5]

        #sematic
        self.n_graph = 5
        self.UAV_image_select = np.zeros((self.n_UAV, self.n_graph))
        self.graph = np.arange((self.n_graph))
        self.triplet_processed = np.asarray([[[   48,  5856],   #gamma = 0.1
                                            [   19,  2360],   #gamma = 0.3
                                            [    6,   776]],  #gamma = 0.5

                                           [[   44,  5344],
                                            [   17,  2136],
                                            [   2,  233]],

                                           [[   34, 4216],
                                            [    19, 2416],
                                            [    1,   104]],

                                           [[   33, 3752],
                                            [    18,   2356],
                                            [    6,   648]],

                                           [[   43, 4800],
                                            [   12, 1352],
                                            [    4,   416]]])

        self.triplet_ori = np.asarray([[ 200, 23368],  #gamma = 0
                                       [ 200, 24736],
                                       [ 200, 23704],
                                       [ 200, 22600],
                                       [ 200, 22880]])


        #MAP
        self.max_x = 250
        self.min_x = -250
        self.max_y = 250
        self.min_y = -250

        # self.UAV_coordinate = np.array([[self.max_x/3*2, self.max_y/5, 100], [self.min_x/3*2, self.max_y/5, 100], [0, self.max_y/4*3, 100], [self.max_x/2, self.min_y/3*2, 100], [self.min_x/2, self.min_y/3*2, 100]])
        self.UAV_coordinate = np.array([[self.max_x/2, 0, 100], [self.min_x/2, 0, 100], [0, self.max_y/2, 100], [0, self.min_y/2, 100]])
        self.BS_coordinate = [0,0,25]
        self.user_coordinate = np.zeros((self.n_user, 3))
        self.distance = np.zeros((self.n_user, self.n_UAV))

    #======================= USER UPLINK =========================
    def renew_user_distance(self):
        x_coordinate = np.random.uniform(-250, 250, self.n_user) # Generate random x and y coordinates
        y_coordinate = np.random.uniform(-250, 250, self.n_user) # Create an array of zeros for the z-coordinates
        z_coordinate = np.zeros((self.n_user))
        self.user_coordinate = np.column_stack((x_coordinate, y_coordinate, z_coordinate)) # Combine the x, y, and z coordinates to create the user coordinates
        for i in range(self.n_user):
          for j in range(self.n_UAV):
              self.distance[i, j] = np.sqrt(np.sum((self.user_coordinate[i] - self.UAV_coordinate[j])**2))
        # print(self.distance)

    def renew_user2UAV_channel(self):
        self.h_u = np.ones((self.n_user, self.n_UAV))
        for i in range(self.n_user):
            for j in range(self.n_UAV):
                self.Gamma_i[i, j] = 10**(np.random.choice(self.Gamma_list) / 10) 
    
    def Compute_Q_data(self): # update the user_UAV with SINR
        SINR_user_UAV = np.zeros((self.n_user, self.n_UAV))
        number_of_user = np.zeros(self.n_UAV)
        self.Q_data = np.zeros((self.n_UAV))
        for i in range(self.n_user):
            Interference = np.zeros((self.n_UAV))
            Signal = np.zeros((self.n_UAV))
            for j in range(self.n_UAV):
                distance = self.distance[i, j]  
    
                Signal[j] = 10**(self.power_user_dB / 10) * self.h_u[i,j] * (distance**(-2))
                Interference[j] = self.Gamma_i[i, j] + self.bandwidthdB_i_outside * self.sig2
            
            # print(Signal, "userSignal")
            # print(Interference, "userInterfere")

            SINR = np.divide(Signal, Interference)
            # print(SINR)
           
            chosen_UAV_index = np.argmax(SINR)
            number_of_user[chosen_UAV_index] += 1
            SINR_user_UAV[i, chosen_UAV_index] = np.max(SINR) 

        # print(SINR_user_UAV, 'userSINR')
        
        for j in range(self.n_UAV):            
            if number_of_user[j] != 0:
                for i in range(self.n_user):
                    if SINR_user_UAV[i,j] != 0:
                        self.Q_data[j] += self.tau1 * self.bandwidthdB_i_outside/number_of_user[j] * np.log2(1 + SINR_user_UAV[i,j]) #update Q_user_UAV
        # print(self.Q_data)

    #==================== UAV-TO-BS =====================
    def renew_BS_channel(self):
        self.h_i = np.abs(np.random.normal(0, 1, (self.n_UAV, self.n_RB))) 

    def Compute_Performance_Reward_Train(self, actions_power):

        actions = actions_power[:, 0]  # the channel_selection_part
        power_selection = actions_power[:, 1]  # power selection

        # ------------ Compute V2V rate -------------------------
        UAV_Interference = np.zeros((self.n_UAV))
        UAV_Signal = np.zeros((self.n_UAV))
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                UAV_Signal[indexes[j, 0]] = 10**(self.power_dB_list[power_selection[indexes[j, 0]]] / 10) * self.h_i[indexes[j, 0], i] * (125**2 + 75**2)**(-1)
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    UAV_Interference[indexes[j, 0]] += 10**(self.power_dB_list[power_selection[indexes[k, 0]]] / 10) * self.h_i[indexes[k, 0], i]
                    UAV_Interference[indexes[k, 0]] += 10**(self.power_dB_list[power_selection[indexes[j, 0]]] / 10) * self.h_i[indexes[j, 0], i]
        # print(V2V_Signal)
        UAV_Interference_final = UAV_Interference + self.bandwidthdB_B_outside * self.sig2
        rate = np.log2(1 + np.divide(UAV_Signal, UAV_Interference_final))

        # print(rate * self.tau2 * self.bandwidthdB_B_outside)
        transfer = rate * self.tau2 * self.bandwidthdB_B_outside

        return transfer, rate

    def Compute_Interference(self, actions):
        UAV_Interference = np.zeros((self.n_UAV, self.n_RB)) + self.bandwidthdB_B_outside *self.sig2

        channel_selection = actions.copy()[:, 0]
        power_selection = actions.copy()[:, 1]

        for i in range(self.n_UAV):
            for k in range(self.n_UAV):
                if i == k:
                    continue
                UAV_Interference[k, channel_selection[i]] += 10 ** (self.power_dB_list[power_selection[i]] / 10) * self.h_i[i, channel_selection[i]]
        self.UAV_Interference_all = 10 * np.log10(UAV_Interference)

    def act_for_training(self, actions):
        
        normalizer = 5
        d = np.ones((self.n_UAV))
        reward = np.zeros((self.n_UAV))
        action_temp = actions.copy()
        gamma_i = actions[:,2]
        transfer, rate = self.Compute_Performance_Reward_Train(action_temp)
        omega1 = 0.004
        omega2 = 0.0004
        omega3 = 31 #a = max 17 ( 17/0.58 floor)
        omega4 = 1 
        beta1 = 68
        
        for i in range(self.n_UAV):
            a = 0
            all_Qgraph = 0
            rho, processed_Qgraph = self.triplet_process(self.UAV_image_select[i, :], gamma_i[i])
            for j in range(len(np.asarray(np.where(self.UAV_image_select[i, :] == 1)).flatten())):
                a +=  (1 - np.exp(np.divide(-rho[0, j], omega4)))
            # print(a) # 0.56 voi omega3 = 1
            all_Qgraph = np.sum(processed_Qgraph)

            # print(transfer[i], '  ',all_Qgraph + self.Q_data[i])
            # print(all_Qgraph)
            # print(rate[i]) #max 17
            
            if transfer[i] < self.Q_data[i]:
                reward[i] += - (omega1 * (self.Q_data[i] - transfer[i]) + omega2 * all_Qgraph)
                # print(-  omega2 * all_Qgraph)
                d[i] = transfer[i] / self.Q_data[i]

            elif transfer[i] < (self.Q_data[i] + all_Qgraph):
                reward[i] += - omega2 * (self.Q_data[i] + all_Qgraph - transfer[i])

            else:
                reward[i] += rate[i] + omega3*a

            if rate[i] + omega3*a > 30:
                reward[i] = beta1
                # print(2)
        reward = np.sum(reward) / (self.n_UAV)
        b = sum(d) / self.n_UAV

        return reward/normalizer, b, d

    def new_random_game(self):
        # make a new game
        self.renew_user_distance()
        self.renew_user2UAV_channel()
        self.Compute_Q_data()
        self.Qgraph_gen()
        self.renew_BS_channel()
    
    #==================== SEMATIC =====================

    def Qgraph_gen(self):
        self.UAV_image_select = np.zeros((self.n_UAV, self.n_graph))
        for i in range(self.n_UAV):
            # n_images = np.random.randint(2, 4)  # Choose number of images
            n_images = 3
            chosen_images_index = np.random.choice(self.graph, n_images, replace=False)  # Choose images
            self.UAV_image_select[i, chosen_images_index] = 1

    def triplet_process(self, UAV_image_select, gamma_i_index):
        chosen_images_index = np.where(UAV_image_select == 1)

        ori_triplet = self.triplet_ori[chosen_images_index, 0]
        # print(ori_triplet)
        processed_triplet = self.triplet_processed[chosen_images_index, gamma_i_index, 0]
        # print(processed_triplet)
        processed_Qgraph = self.triplet_processed[chosen_images_index, gamma_i_index, 1]
        rho = np.divide(processed_triplet, ori_triplet)
        return rho, processed_Qgraph

    def get_ori_Qgraph(self, UAV_image_select):
        chosen_images_index = np.where(UAV_image_select == 1)

        all_ori_Qgraph = 0
        ori_Qgraph = self.triplet_ori[chosen_images_index, 1]
        n_images = len(self.triplet_ori[chosen_images_index, 1][0])
        # print(n_images)
        all_ori_Qgraph = np.sum(ori_Qgraph)
        return all_ori_Qgraph, n_images






