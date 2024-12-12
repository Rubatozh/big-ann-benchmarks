import PyCANDY
import time
import numpy as np
import collections
from typing import List
import torch
#from neurips23.streaming.dagnn.SAFERL.CPQ import CPQ
from .SAFERL.CPQ import CPQ
from neurips23.streaming.base import BaseStreamingANN


class dagnn(BaseStreamingANN):
    def __init__(self, metric, index_params):
        self.name="dagnn"
        self.is_config = False
        self.lr = 2e-3
        self.num_episodes = 500
        self.hidden_dim = 128
        self.gamma = 0.98
        self.epsilon = 0.01
        self.target_update = 10
        self.minimal_size = 100
        self.sample_batch_size = 64
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        self.state_dim =21
        self.action_dim = 1
        self.max_action = 1
        policy = CPQ(self.state_dim, self.action_dim, self.max_action, discount=0.99,threshold=-0.95, alpha=10)


        import os
        cwd = os.getcwd()
        print(cwd)

        #policy.actor =  torch.load("neurips23/streaming/dagnn/models/actor"+"0.0"+".pt",map_location=torch.device('cpu'))
        #policy.cost_critic=torch.load("neurips23/streaming/dagnn/models/cost_critic"+"0.0"+".pt",map_location=torch.device('cpu'))
        #policy.cost_critic_target=torch.load("neurips23/streaming/dagnn/models/cost_critic_target"+"0.0"+".pt",map_location=torch.device('cpu'))
        #policy.reward_critic=torch.load( "neurips23/streaming/dagnn/models/reward_critic"+"0.0"+".pt",map_location=torch.device('cpu'))
        #policy.reward_critic_target = torch.load( "neurips23/streaming/dagnn/models/reward_critic_target"+"0.0"+".pt",map_location=torch.device('cpu'))
        #policy.vae=torch.load("neurips23/streaming/dagnn/models/vae"+"0.0"+".pt",map_location=torch.device('cpu'))

        self.policy = policy

        self._index_params = index_params

        self.index = PyCANDY.DAGNNIndex()






    def encode_states_to_tensor(self, state):
        gs = state.global_stat
        bs = state.time_local_stat
        ws = state.window_states

        combined_gs = [gs.degree_sum/(bs.ntotal+bs.old_ntotal) if bs.ntotal+bs.old_ntotal!=0 else 0, gs.degree_variance, gs.neighbor_distance_sum/(bs.ntotal+bs.old_ntotal) if bs.ntotal+bs.old_ntotal!=0 else 0, gs.neighbor_distance_variance, gs.steps_expansion_average, gs.steps_taken_avg, gs.steps_taken_max]
        combined_bs = [bs.degree_sum_new/bs.ntotal if bs.ntotal!=0 else 0, bs.degree_sum_old/bs.old_ntotal if bs.old_ntotal!=0 else 0, bs.degree_variance_new, bs.degree_variance_old, bs.neighbor_distance_sum_new/bs.ntotal if bs.ntotal!=0 else 0, bs.neighbor_distance_sum_old/bs.old_ntotal if bs.old_ntotal!=0 else 0, bs.neighbor_distance_variance_new, bs.neighbor_distance_variance_old, bs.steps_expansion_sum/bs.ntotal if bs.ntotal!=0 else 0, bs.steps_taken_max, bs.steps_taken_sum/bs.ntotal if bs.ntotal!=0 else 0]
        combined_ws = [ ws.getCount(0), ws.getCount(1), ws.getCount(2)]

        combined_state = combined_gs+combined_bs+combined_ws
        state_tensor = torch.tensor(combined_state, dtype=torch.float)
        return state_tensor

    def insert(self, X, ids):


        raw_state = self.index.getState()
        state = self.encode_states_to_tensor(raw_state)
        action = (int)(self.policy.select_action(state)*8)
        self.index.performAction(action)
        subA = torch.from_numpy(X.copy())


        self.index.insertTensor(subA)



        return

    def setup(self, dtype, max_pts, ndims):
        self.vecDim =  ndims
        cfg = PyCANDY.ConfigMap()
        cfg.edit("vecDim", self.vecDim)
        self.index.setConfig(cfg)


    def delete(self, ids):

        return

    def query(self, X, k):

        queryTensor = torch.from_numpy(X.copy())
        results = self.index.searchIndex(queryTensor, k)
        res = np.array(results).reshape(X.shape[0], k)


        self.res = res

