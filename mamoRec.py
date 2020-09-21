# Author: Manqing Dong, 2020
import torch
from modules.input_loading import MLUserLoading, MLItemLoading, BKUserLoading, BKItemLoading
from modules.info_embedding import ItemEmbedding, UserEmbedding
from modules.rec_model import RecMAM
from modules.memories import FeatureMem, TaskMem
from models import BASEModel, LOCALUpdate, maml_train, user_mem_init
from configs import config_settings
from utils import *
from tqdm import tqdm


class MAMRec:
    def __init__(self, dataset='movielens'):

        self.dataset = dataset
        self.support_size = config_settings['support_size'] # 15
        self.query_size = config_settings['query_size'] # 5
        self.n_epoch = config_settings['n_epoch'] # 3
        self.n_inner_loop = config_settings['n_inner_loop'] # 3
        self.batch_size = config_settings['batch_size'] # 5
        self.n_layer = config_settings['n_layer'] # 2
        self.embedding_dim = config_settings['embedding_dim'] # 100
        self.lr_in = config_settings['lr_in']  # local learning rate, 0.01
        self.lr_out = config_settings['lr_out']  # global learning rate, 0.05
        self.tau = config_settings['tau']  # hyper-parameter for initializing personalized u weights, 0.01
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config_settings['cuda_option'] if self.USE_CUDA else "cpu")
        self.n_k = config_settings['n_k'] # 3
        self.alpha = config_settings['alpha'] # 0.5
        self.beta = config_settings['beta'] # 0.05
        self.gamma = config_settings['gamma'] # 0.1
        self.active_func = config_settings['active_func'] # leaky_relu
        self.rand = config_settings['rand'] # True
        self.random_state = config_settings['random_state'] # 100
        self.split_ratio = config_settings['split_ratio'] # 0.8

        # load dataset
        print('Start initializing...')
        # train_users: 4832, test_users: 1207
        self.train_users, self.test_users = train_test_user_list(dataset=dataset, rand=self.rand,
                                                                 random_state=self.random_state,
                                                                 train_test_split_ratio=self.split_ratio)

        if dataset == 'movielens':
            self.x1_loading, self.x2_loading = MLUserLoading(embedding_dim=self.embedding_dim).to(self.device), \
                                               MLItemLoading(embedding_dim=self.embedding_dim).to(self.device)
        else:
            self.x1_loading, self.x2_loading = BKUserLoading(embedding_dim=self.embedding_dim).to(self.device), \
                                               BKItemLoading(embedding_dim=self.embedding_dim).to(self.device)

        self.n_y = default_info[dataset]['n_y']

        # Embedding model
        self.UEmb = UserEmbedding(self.n_layer, default_info[dataset]['u_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)
        self.IEmb = ItemEmbedding(self.n_layer, default_info[dataset]['i_in_dim'] * self.embedding_dim,
                                  self.embedding_dim, activation=self.active_func).to(self.device)

        # rec model
        self.rec_model = RecMAM(self.embedding_dim, self.n_y, self.n_layer, activation=self.active_func).to(self.device)

        # whole model
        self.model = BASEModel(self.x1_loading, self.x2_loading, self.UEmb, self.IEmb, self.rec_model).to(self.device)

        self.phi_u, self.phi_i, self.phi_r = self.model.get_weights()

        self.FeatureMEM = FeatureMem(self.n_k, default_info[dataset]['u_in_dim'] * self.embedding_dim,
                                     self.model, device=self.device)
        self.TaskMEM = TaskMem(self.n_k, self.embedding_dim, device=self.device)
        
        print('Initializing finished.')

        self.train = self.train_with_meta_optimization
        self.test = self.test_with_meta_optimization


    def train_with_meta_optimization(self):
        for i in range(self.n_epoch):
            u_grad_sum, i_grad_sum, r_grad_sum = self.model.get_zero_weights()
            loss_list = []

            # On training dataset
            for u in tqdm(self.train_users[:100]):
                # init local parameters: theta_u, theta_i, theta_r
                bias_term, att_values = user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading, self.alpha)
                self.model.init_u_mem_weights(self.phi_u, bias_term, self.tau, self.phi_i, self.phi_r)
                self.model.init_ui_mem_weights(att_values, self.TaskMEM)

                user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                          self.n_inner_loop, self.lr_in, top_k=3, device=self.device)
                u_grad, i_grad, r_grad, loss = user_module.train()
                loss_list.append(loss)

                u_grad_sum, i_grad_sum, r_grad_sum = grads_sum(u_grad_sum, u_grad), grads_sum(i_grad_sum, i_grad), \
                                                     grads_sum(r_grad_sum, r_grad)

                self.FeatureMEM.write_head(u_grad, self.beta)
                u_mui = self.model.get_ui_mem_weights()
                self.TaskMEM.write_head(u_mui[0], self.gamma)

            print('mean_loss: {}'.format(sum(loss_list) / len(loss_list)))
            self.phi_u, self.phi_i, self.phi_r = maml_train(self.phi_u, self.phi_i, self.phi_r,
                                                            u_grad_sum, i_grad_sum, r_grad_sum, self.lr_out)

            # self.test_with_meta_optimization()

    def test_with_meta_optimization(self):
        best_phi_u, best_phi_i, best_phi_r = self.model.get_weights()

        for u in self.test_users:
            bias_term, att_values = user_mem_init(u, self.dataset, self.device, self.FeatureMEM, self.x1_loading, self.alpha)
            self.model.init_u_mem_weights(best_phi_u, bias_term, self.tau, best_phi_i, best_phi_r)
            self.model.init_ui_mem_weights(att_values, self.TaskMEM)

            self.model.init_weights(best_phi_u, best_phi_i, best_phi_r)
            user_module = LOCALUpdate(self.model, u, self.dataset, self.support_size, self.query_size, self.batch_size,
                                      self.n_inner_loop, self.lr_in, top_k=3, device=self.device)
            user_module.test()


if __name__ == '__main__':
    mam_rec = MAMRec('movielens')
    mam_rec.train()