import os
from time import time
from collections import deque
import random
import numpy as np
import sys
import argparse
import math
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.envs.toy_env.toy_env import ToyEnv
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *
from scipy.stats import multivariate_normal
from sklearn.linear_model import Ridge

class graph_model():
    def __init__(self, data, obs_dim, action_dim, num_stage, enable_faulty_state = False, enable_reverse_prob = False, learning_rate = 1e-4, gradient_clip = 10):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.enable_faulty_state =  enable_faulty_state
        self.lr = learning_rate
        self.gd_threshold = 0.1
        self.gd_clip = gradient_clip
        self.last_max_pos = 0
        self.reverse = enable_reverse_prob

        amtx = np.zeros((obs_dim,obs_dim))
        np.fill_diagonal(amtx,1)
        #self.A_lst = [np.random.rand(obs_dim*obs_dim).reshape((obs_dim,obs_dim)) for i in range(num_stage+1)]
        self.A_lst = [amtx for i in range(num_stage + 1)]
        bmtx = np.zeros((obs_dim,action_dim))
        #self.B_lst = [np.random.rand(obs_dim*action_dim).reshape((obs_dim,action_dim)) for i in range(num_stage+1)]
        self.B_lst = [bmtx for i in range(num_stage + 1)]
        self.Wij_lst = np.zeros((num_stage+1, num_stage+1, obs_dim))
        for idx in range(num_stage+1):
            self.Wij_lst[idx,idx,:] = np.random.rand(obs_dim)
        for idx in range(num_stage - 1): #forward progress
            self.Wij_lst[idx, idx + 1, :] = np.random.rand(obs_dim)

        if enable_faulty_state:
            self.W0_lst = [np.random.rand(obs_dim) for i in range(num_stage+1)]
            for idx in range(num_stage + 1):
                self.Wij_lst[idx, num_stage, :] = np.random.rand(obs_dim)
            self.Wij_lst[num_stage-1, num_stage, :] = np.random.rand(obs_dim)
            #first num_stage :  Wii (n+1), Wif (n), Wij (n-1)

        if enable_reverse_prob:
            for idx in range(num_stage-1):
                self.Wij_lst[idx+1, idx , :] = np.random.rand(obs_dim) #first num_stage :  Wii (n+1), Wif (n), Wij (n-1), Wji(n-1)
        self.sigma_lst = []
        for i in range(num_stage+1):
            A = np.random.rand(obs_dim, obs_dim)
            amtx = np.zeros((obs_dim, obs_dim))
            np.fill_diagonal(amtx, 1)
            self.sigma_lst.append(amtx)

        self.data = data
        self.data_normalized = None
        self.num_stage = num_stage
        self.Pj = [] #store Pt=j | Z 1:N data for each sequence

    def logit(self, state, state_idx):
        Pijs = np.zeros((self.num_stage+1, self.num_stage+1))
        if state_idx == 0:
            #for idj, w in enumerate(self.W0_lst):
            #    Pijs[idj,idj] = np.matmul(w.T, state)
            #Pijs/=np.sum(np.sum(Pijs))
            Pijs[0, 0] = 1
            Pijs[1, 1] = 1
            #Pijs[2, 2] = 1
            Pijs /= np.sum(np.sum(Pijs))
        else:
            for idi in range(self.num_stage+1):
                for idj in range(self.num_stage + 1):
                    if np.matmul(self.Wij_lst[idi,idj,:], state)!=0:
                        Pijs[idi, idj] = np.exp(np.matmul(self.Wij_lst[idi,idj,:], state))
            #normalize
            for idj in range(self.num_stage+1): #last row always empty
                if np.sum(Pijs[idj, :]) !=0:
                    Pijs[idj, :] /= np.sum(Pijs[idj, :])
        return Pijs
    def trans_prob(self, s, a, sn):
        trans_prob = np.zeros(self.num_stage+1)
        for idx in range(self.num_stage+1):
            rv = multivariate_normal(np.matmul(self.A_lst[idx], s)+np.matmul(self.B_lst[idx], a),
                                     self.sigma_lst[idx], allow_singular=True)
            trans_prob[idx]=rv.pdf(sn)
        return trans_prob

    def normalize(self):
        obs_aggregated = []
        action_aggregated = []
        for seq in self.data:
            for obs in seq[0]:
                obs_aggregated.append(obs[0])
                action_aggregated.append(obs[1])
        obs_aggregated = np.array(obs_aggregated)
        action_aggregated = np.array(action_aggregated)
        self.obs_mean = np.mean(obs_aggregated, axis=0)
        self.obs_std = np.std(obs_aggregated, axis=0)
        self.action_mean = np.mean(action_aggregated, axis=0)
        self.action_std = np.std(action_aggregated, axis=0)
        self.data_normalized = []
        #print(self.obs_mean,self.obs_std)
        for seq in self.data:
            new_seq=[]
            for obs in seq[0]:
                #print(obs[0], obs[1], obs[3])
                a = (obs[0] - self.obs_mean)/self.obs_std
                b = (obs[1] - self.action_mean)/self.action_std
                c = (obs[3] - self.obs_mean)/self.obs_std
                #print(a, c)
                new_seq.append((a,b,obs[2], c))
            self.data_normalized.append(new_seq)

    def forward_passing(self):
        Pj_zf_ens = []
        for sample_seq in self.data_normalized:
            a_seq = np.zeros((self.num_stage + 1, len(sample_seq))) + 1e-60
            Pj_zf = np.zeros((self.num_stage + 1, len(sample_seq))) + 1e-60
            for ids, obs in enumerate(sample_seq):
                Pijs = self.logit(obs[0], ids)
                trans_pb = self.trans_prob(obs[0],obs[1],obs[3])*np.exp(obs[2]) + 1e-60
                if ids == 0: #is first sample
                    for idstg in range(self.num_stage+1):
                        a_seq[idstg, ids] = trans_pb[idstg]*Pijs[idstg,idstg]
                        Pj_zf[idstg, ids] = Pijs[idstg,idstg]
                else:
                    for idstg in range(self.num_stage + 1):
                        for idstgi in range(self.num_stage + 1):
                            Pj_zf[idstg, ids] +=  a_seq[idstgi, ids-1]*Pijs[idstgi,idstg]
                            a_seq[idstg, ids] += trans_pb[idstg]*a_seq[idstgi, ids-1]*Pijs[idstgi,idstg]
                #normalization doesn't affect Pjz value:
                #if np.argmax(Pj_zf[:, ids]) < self.last_max_pos and ids>0:
                #    print(Pj_zf)
                self.last_max_pos = np.argmax(Pj_zf[:, ids])
                if np.sum(a_seq[:, ids]) == 0:
                    print(a_seq[:, ids])
                #a_seq[:, ids] /= np.sum(a_seq[:, ids])
            print(np.argmax(Pj_zf, axis = 0))
            Pj_zf_ens.append(np.argmax(Pj_zf, axis = 0))
        #Pj_zf_ens = np.array(Pj_zf_ens)
        return Pj_zf_ens
    def info_passing(self):
        Pj_z_ens = []
        Pij_z_ens_all = []
        Pijs_ens_all = []
        for sample_seq in self.data_normalized:
            a_seq = np.zeros((self.num_stage+1, len(sample_seq))) + 1e-60
            b_seq = np.zeros((self.num_stage+1, len(sample_seq))) + 1e-60
            #buffer of transition probabilities over all episodes, avoid recalculation in back probagation
            Pijs_ens = []
            Pij_z_ens = []
            trans_pb_ens = []
            Pj_z = np.zeros((self.num_stage+1, len(sample_seq)))
            for ids, obs in enumerate(sample_seq):
                Pijs = self.logit(obs[0], ids)
                trans_pb = self.trans_prob(obs[0],obs[1],obs[3])*np.exp(obs[2]) + 1e-60
                Pijs_ens.append(Pijs)
                trans_pb_ens.append(trans_pb)
                if ids == 0: #is first sample
                    for idstg in range(self.num_stage+1):
                        a_seq[idstg, ids] = trans_pb[idstg]*Pijs[idstg,idstg]
                else:
                    for idstg in range(self.num_stage + 1):
                        for idstgi in range(self.num_stage + 1):
                            a_seq[idstg, ids] += trans_pb[idstg]*a_seq[idstgi, ids-1]*Pijs[idstgi,idstg]
                #normalization doesn't affect Pjz value:
                if np.sum(a_seq[:, ids]) == 0:
                    print(a_seq[:, ids])
                #a_seq[:, ids] /= np.sum(a_seq[:, ids])
            #print(a_seq)
            for ids in range(len(sample_seq)):
                idsn=len(sample_seq)-ids-1
                if ids == 0:
                    b_seq[:, idsn] = 1
                else:
                    for idstg in range(self.num_stage + 1):
                        for idstgi in range(self.num_stage + 1):
                            Pijs_t = Pijs_ens[idsn+1]
                            trans_pb_t = trans_pb_ens[idsn+1]
                            b_seq[idstg, idsn] += Pijs_t[idstg, idstgi] * trans_pb_t[idstgi] * b_seq[idstgi, idsn+1]
                #b_seq[:, idsn] /= np.sum(b_seq[:, idsn])
            for ids in range(len(sample_seq)):
                Pj_z[:, ids] = (a_seq[:, ids]*b_seq[:,ids])/ np.sum(a_seq[:, ids]*b_seq[:,ids])
            Pj_z_ens.append(Pj_z)
            for ids in range(len(sample_seq)-1):
                Pij_z = np.zeros((self.num_stage + 1, self.num_stage + 1))
                for idstgj in range(self.num_stage + 1):
                    for idstgi in range(self.num_stage + 1):
                        Pijs_t = Pijs_ens[ids+1]
                        trans_pb_t = trans_pb_ens[ids+1]
                        Pij_z[idstgi, idstgj] = a_seq[idstgi, ids] * Pijs_t[idstgi, idstgj] * \
                                                trans_pb_t[idstgj] * b_seq[idstgj, ids+1]
                        Pij_z[idstgi, idstgj]/=np.sum(a_seq[:, ids]*b_seq[:,ids])
                        #if Pij_z[idstgi, idstgj]>1:
                        #    print(Pij_z)
                Pij_z_ens.append(Pij_z)
            Pij_z_ens_all.append(Pij_z_ens)
            Pijs_ens_all.append(Pijs_ens)

        return Pj_z_ens, Pij_z_ens_all , Pijs_ens_all

    def idx_to_wij_seq(self, i, j):
        if self.enable_faulty_state:
            if i == j:
                return i
            if i < j and j == self.num_stage:
                return self.num_stage + 1 + i
            if i == j-1 and j < self.num_stage:
                return 2* self.num_stage + 1 + i
        else:
            if i == j and j < self.num_stage:
                return i
            if i == j - 1 and j < self.num_stage:
                return 2* self.num_stage + 1 + i
        if self.reverse:
            if j == i - 1 and i < self.num_stage:
                return 1

        return -1

    def update_param(self, Pj_z_ens, Pij_z_ens_all , Pijs_ens_all):
        if self.enable_faulty_state:
            stage_itr = self.num_stage + 1
        else:
            stage_itr = self.num_stage

        for idc in range(stage_itr):
            Y=[]
            X=[]
            W=[]
            itemcount = 0
            for idsamp, sample_seq in enumerate(self.data_normalized):
                for ids, obs in enumerate(sample_seq):
                    Y.append(obs[3])
                    X.append(np.concatenate((obs[0], obs[1]),axis=0))
                    W.append(Pj_z_ens[idsamp][idc, ids])
                    itemcount+=1
            Y = np.array(Y).T
            X = np.array(X).T
            W = np.diag(W)
            tmp_inv = np.linalg.inv(X.dot(W).dot(X.T))
            tmp = Y.dot(W).dot(X.T).dot(tmp_inv)
            #update linear params
            self.A_lst[idc] = tmp[:,:self.obs_dim]
            self.B_lst[idc] = tmp[:, self.obs_dim:(self.obs_dim+self.action_dim)]

            #update std
            Psum = np.sum(W.diagonal())
            sigma = np.zeros((self.obs_dim, self.obs_dim))
            sample_count=0
            for idsamp, sample_seq in enumerate(self.data_normalized):
                for ids, obs in enumerate(sample_seq):
                    #print(ids, np.argmax(Pj_z_ens[idsamp][:, ids]))
                    next_obs_est = np.matmul(self.A_lst[idc], obs[0])+np.matmul(self.B_lst[idc], obs[1])
                    sigma += Pj_z_ens[idsamp][idc, ids]*np.dot((obs[3]-next_obs_est)[:,None], (obs[3]-next_obs_est)[None,:])
                    #sigma += np.dot((obs[3]-next_obs_est)[:,None], (obs[3]-next_obs_est)[None,:])
                    sample_count+=1
            #sigma/=sample_count
            if Psum!=0:
                sigma/=Psum
            #print(sigma)
            self.sigma_lst[idc] = sigma

        #update Wij
        S = []

        for idsamp, sample_seq in enumerate(self.data_normalized):
            for ids in range(len(sample_seq)-2): #from no.2 to n-1
                S.append(sample_seq[ids+1][0])
        S=np.array(S)
        #for each pair of state transitions
        for idi in range(self.num_stage + 1):
            #Ptclswt = []
            for idj in range(self.num_stage + 1):
                Ptclswj = []
                widx = self.idx_to_wij_seq(idi, idj)
                if widx != -1:
                    L = []
                    P = []
                    for idsamp, sample_seq in enumerate(self.data_normalized):
                        Pijs_ens = Pijs_ens_all[idsamp]
                        Pij_z_ens = Pij_z_ens_all[idsamp]
                        for ids in range(len(sample_seq) - 2):  # from no.2 to n-1
                            if Pj_z_ens[idsamp][idi, ids] == 0  or Pij_z_ens[ids][idi,idj] == 0:
                                Ptclswj.append(0)
                            else:
                                Ptclswj.append(np.log(Pij_z_ens[ids][idi,idj]/Pj_z_ens[idsamp][idi, ids]))
                            L.append(Pij_z_ens[ids][idi,idj])
                            P.append(Pijs_ens[ids + 1][idi,idj])
                    L=np.array(L)
                    P=np.array(P)
                    LR_target = np.array(Ptclswj).T
                    G = (S.T).dot(P-L)
                    print(G, np.linalg.norm(G), idi, idj)
                    clf = Ridge(alpha=0.5)
                    clf.fit(S, LR_target)
                    if np.linalg.norm(G)>self.gd_clip:
                        G = self.gd_clip*(G/np.linalg.norm(G))
                    #self.Wij_lst[idi,idj,:] -= self.lr * G
                    self.Wij_lst[idi,idj,:] = clf.coef_
                #if len(Ptclswj) >0:
                #    Ptclswt.append(Ptclswj)
            #LR_target = np.array(Ptclswt).T
            #clf = LogisticRegression(random_state=0).fit(S, np.argmax(LR_target, axis = 1))
            #print(len(Ptclswt))









def main():
    #load parameters
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    #manual input of arguments tobe removed finally.
    args.eval_only=True
    args.experiment_folder = '/media/zhq/A4AB-0E37/igibson_FDD_exp_lowdim_onestage'
    with open(os.path.join(args.experiment_folder, 'bfdd_sas_without_rgb_1.pkl'), 'rb') as handle:
        collected_fdd_list = pickle.load(handle)
    #args.experiment_folder = '/media/zhq/A4AB-0E37/igibson_FDD_exp'
    #with open(os.path.join(args.experiment_folder, 'fdd_sas_with_rgb.pkl'), 'rb') as handle:
    #    collected_fdd_list = pickle.load(handle)
    print(len(collected_fdd_list))
    np.random.seed(1)
    gm = graph_model(collected_fdd_list, 6, 7, 2)
    gm.normalize()
    this_result=[]
    for idx in range(50):
        Pjz_ens, Pij_z_ens_all, Pijs_ens_all = gm.info_passing()
        gm.update_param(Pjz_ens, Pij_z_ens_all, Pijs_ens_all)
        this_result = gm.forward_passing()

    with open(os.path.join(args.experiment_folder, 'bfdd_sas_without_rgb_1_seg_res.pkl'), 'wb') as handle:
        pickle.dump(this_result, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    main()