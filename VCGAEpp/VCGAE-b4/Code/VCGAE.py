
import copy
import tensorflow as tf
import os
import sys
import random
import logging
from datetime import datetime
import numpy as np
from tensorflow.contrib.layers import xavier_initializer



from utility.helper import *
from utility.batch_test import *

np.random.seed(2022)
random.seed(2022)
tf.set_random_seed(2022)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
class LightGCN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.pv_adj = data_config['pv_adj']
        self.cart_adj = data_config['cart_adj']
        self.fav_adj = data_config['fav_adj']

        self.pv_mat = config['pv_mat']
        self.fav_mat = config['fav_mat']
        self.cart_mat = config['cart_mat']
        self.buy_mat = config['buy_mat']

        self.ssl_temp = args.ssl_temp
        self.tradeOff_ssl = args.tradeOff_ssl

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir=self.create_model_str()
        self.verbose = args.verbose

        self.tradeOff = args.tradeOff
        self.tradeOff_cr = args.tradeOff_cr

        self.n_nodes = self.n_users + self.n_items
        self.n_relations=4
        self.att_head = 2


        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)



        self.weights = self._init_weights()



        all_embeddings_pv = self._create_pv_embed()
        if self.node_dropout_flag:
            all_embeddings_pv =  tf.nn.dropout(all_embeddings_pv, 1 - self.mess_dropout[0])
        self.all_embeddings_pv = all_embeddings_pv
        self.all_embeddings_pv_all = all_embeddings_pv
        self.ua_embeddings_pv, self.ia_embeddings_pv = tf.split(self.all_embeddings_pv, [self.n_users, self.n_items], 0)


        all_embeddings_fav = self._create_fav_embed()
        if self.node_dropout_flag:
            all_embeddings_fav = tf.nn.dropout(all_embeddings_fav, 1 - self.mess_dropout[0])
        self.all_embeddings_fav = all_embeddings_fav
        self.all_embeddings_fav_all = all_embeddings_fav + self.all_embeddings_pv_all
        self.ua_embeddings_fav, self.ia_embeddings_fav = tf.split(self.all_embeddings_fav, [self.n_users, self.n_items],0)


        all_embeddings_cart = self._create_cart_embed()
        if self.node_dropout_flag:
            all_embeddings_cart =  tf.nn.dropout(all_embeddings_cart, 1 - self.mess_dropout[0])
        self.all_embeddings_cart = all_embeddings_cart
        self.all_embeddings_cart_all = all_embeddings_cart  +  self.all_embeddings_pv_all  +  self.all_embeddings_fav_all
        self.ua_embeddings_cart, self.ia_embeddings_cart = tf.split(self.all_embeddings_cart, [self.n_users, self.n_items], 0)


        all_embeddings = self._create_lightgcn_embed()
        if self.node_dropout_flag:
            all_embeddings =  tf.nn.dropout(all_embeddings, 1 - self.mess_dropout[0])
        self.all_embeddings_buy = all_embeddings + self.all_embeddings_cart_all + self.all_embeddings_pv_all + self.all_embeddings_fav_all
        self.ua_embeddings_buy, self.ia_embeddings_buy = tf.split(self.all_embeddings_buy, [self.n_users, self.n_items], 0)

        self.global_embeddings = self._create_global_embed()
        if self.node_dropout_flag:
            self.global_embeddings = tf.nn.dropout(self.global_embeddings, 1 - self.mess_dropout[0])
        self.ua_embeddings_glo, self.ia_embeddings_glo = tf.split(self.global_embeddings, [self.n_users, self.n_items],
                                                                  0)



        self.mu_u = self.ua_embeddings_buy +  self.ua_embeddings_glo
        self.mu_i = self.ia_embeddings_buy +  self.ia_embeddings_glo




        logsig_u =  tf.concat([self.ua_embeddings_pv , self.ua_embeddings_fav, self.ua_embeddings_cart],axis=1)
        self.logsig_u =  tf.contrib.layers.fully_connected(logsig_u, self.emb_dim,
                                                         activation_fn=tf.nn.leaky_relu, scope="logsig_u")
        logsig_i =   tf.concat([self.ia_embeddings_pv  ,  self.ia_embeddings_fav, self.ia_embeddings_cart],axis=1)
        self.logsig_i =  tf.contrib.layers.fully_connected(logsig_i, self.emb_dim,
                                                         activation_fn=tf.nn.leaky_relu, scope="logsig_i")



        self.std_u = tf.exp(0.5 * self.logsig_u)
        self.std_i = tf.exp(0.5 * self.logsig_i)
        epsilon_u = tf.random_normal(tf.shape(self.std_u))
        self.z_u_global = self.mu_u + self.is_training_ph * epsilon_u * self.std_u
        epsilon_i = tf.random_normal(tf.shape(self.std_i))
        self.z_i_global = self.mu_i + self.is_training_ph * epsilon_i * self.std_i





        self.u_g_embeddings = tf.nn.embedding_lookup(self.z_u_global, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.z_i_global, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.z_i_global, self.neg_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)


        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)



        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)

        # ----------------------------------Cl loss----------------
        batch_ssl_loss_list = []
        aux_beh_ssl2_loss1 = self.cal_ssloss(self.users, self.buy_mat, self.pv_mat, self.ia_embeddings_buy,
                                             self.ia_embeddings_pv, \
                                             self.weights['projection1'], self.weights['bias1'],
                                             self.weights['projection2'], self.weights['bias2'])
        batch_ssl_loss_list.append(aux_beh_ssl2_loss1)
        aux_beh_ssl2_loss2 = self.cal_ssloss(self.users, self.buy_mat, self.cart_mat, self.ia_embeddings_buy,
                                             self.ia_embeddings_cart, \
                                             self.weights['projection3'], self.weights['bias3'],
                                             self.weights['projection4'], self.weights['bias4'])
        batch_ssl_loss_list.append(aux_beh_ssl2_loss2)
        aux_beh_ssl2_loss3 = self.cal_ssloss(self.users, self.buy_mat, self.fav_mat, self.ia_embeddings_buy,
                                             self.ia_embeddings_fav, \
                                             self.weights['projection5'], self.weights['bias5'],
                                             self.weights['projection6'], self.weights['bias6'])
        batch_ssl_loss_list.append(aux_beh_ssl2_loss3)

        # aux_beh_ssl2_loss4 = self.cal_ssloss(self.users, self.cart_mat, self.pv_mat,  self.ia_embeddings, self.ia_embeddings_pv)
        # batch_ssl_loss_list.append(aux_beh_ssl2_loss4)
        # aux_beh_ssl2_loss5 = self.cal_ssloss(self.users, self.cart_mat, self.fav_mat,  self.ia_embeddings, self.ia_embeddings_fav)
        # batch_ssl_loss_list.append(aux_beh_ssl2_loss5)
        #
        # aux_beh_ssl2_loss6 = self.cal_ssloss(self.users, self.fav_mat, self.pv_mat,  self.ia_embeddings, self.ia_embeddings_pv)
        # batch_ssl_loss_list.append(aux_beh_ssl2_loss6)

        self.batch_ssl_loss = sum(batch_ssl_loss_list)

        self.kl_loss = self.create_kl_loss()
        cr_loss_1 = self.cal_crloss(self.ua_embeddings_buy, self.ua_embeddings_cart, self.ua_embeddings_pv)
        cr_loss_2 = self.cal_crloss(self.ua_embeddings_buy, self.ua_embeddings_fav, self.ua_embeddings_pv)
        cr_loss_3 = self.cal_crloss(self.ua_embeddings_buy, self.ua_embeddings_cart, self.ua_embeddings_fav)
        cr_loss_all =  cr_loss_1 +  cr_loss_2 + cr_loss_3




        self.loss =  (self.mf_loss) + self.tradeOff * self.kl_loss  + (self.tradeOff_cr *  cr_loss_all) +   self.tradeOff_ssl * self.batch_ssl_loss





        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        log_dir = './Log/' + args.dataset + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.%f')
        self.logger = logging.getLogger('Log')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            filename=os.path.join(log_dir,
                                  "VCGAE_%s_batch%d_lr%.4f-%s.res" % (
                                      args.dataset, args.batch_size,
                                      args.lr, timestamp)), mode='w')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        self.logger.info(args)

    def _create_global_embed(self):
        # Generate a set of adjacency sub-matrix.

        self.global_adj = self.norm_adj + self.pv_adj + self.cart_adj + self.fav_adj

        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.global_adj)
        else:
            A_fold_hat = self._split_A_hat(self.global_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # side_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        # return u_g_embeddings, i_g_embeddings
        return all_embeddings

    def cal_ssloss(self,u_batch, tgt_mat, aux_mat, tgt_embeddings_i, aux_embeddings_i, w1,b1,w2,b2):
        ssl2_loss = 0.


        tgt_embeddings_i = tf.add( tf.matmul(tgt_embeddings_i, w1), b1)
        aux_embeddings_i = tf.add( tf.matmul(aux_embeddings_i, w2), b2)

        #[user_num, item_num]
        tgt_mat = self._convert_sp_mat_to_sp_tensor(tgt_mat)
        aux_mat = self._convert_sp_mat_to_sp_tensor(aux_mat)
        #
        tgt_mat = tf.sparse.reorder(tgt_mat)
        aux_mat = tf.sparse.reorder(aux_mat)
        #
        tgt_mat = tf.sparse_tensor_to_dense(tgt_mat)
        aux_mat = tf.sparse_tensor_to_dense(aux_mat)

        emb_aux_batch = tf.gather(aux_mat, u_batch)
        emb_aux = tf.matmul(emb_aux_batch, aux_embeddings_i)  # b,m m,d = b,d
        count = tf.count_nonzero(emb_aux_batch, 1)
        count = tf.expand_dims(count, axis=1)   # b,1
        emb_aux = tf.div_no_nan(emb_aux, tf.cast(count, tf.float32))




        emb_tgt_batch = tf.gather(tgt_mat, u_batch)
        emb_tgt = tf.matmul(emb_tgt_batch, tgt_embeddings_i)  # b,m m,d = b,d
        count = tf.count_nonzero(emb_tgt_batch, 1)
        count = tf.expand_dims(count, axis=1)    # b,1
        emb_tgt = tf.div_no_nan(emb_tgt, tf.cast(count, tf.float32))




        all_emb_aux = tf.matmul(aux_mat, aux_embeddings_i)  # n,m m,d = n,d
        count = tf.count_nonzero(aux_mat, 1)
        count = tf.expand_dims(count, axis=1)   # n,1
        all_emb_aux = tf.div_no_nan(all_emb_aux, tf.cast(count, tf.float32))

        # all_emb_tgt = tf.matmul(tgt_mat, tgt_embeddings_i)  # n,m m,d = n,d
        # count = tf.count_nonzero(tgt_mat, 1)
        # count = tf.expand_dims(count, axis=1)   # n,1
        # all_emb_tgt = tf.div_no_nan(all_emb_tgt, tf.cast(count, tf.float32))
        # all_emb_aux = all_emb_tgt



        self.emb_tgt = tf.nn.l2_normalize(emb_tgt, axis=1)
        self.emb_aux = tf.nn.l2_normalize(emb_aux, axis=1)
        aux_embeddings_u = tf.nn.l2_normalize(all_emb_aux, axis=1)

        pos_score =  tf.reduce_sum(tf.multiply(self.emb_tgt, self.emb_aux), axis=1)
        ttl_score =  tf.matmul(self.emb_tgt, tf.transpose(aux_embeddings_u))     # B*m

        self.pos_score = tf.exp(pos_score / self.ssl_temp)      # B*1
        self.ttl_score = tf.reduce_sum(tf.exp(ttl_score / self.ssl_temp), axis=1)    # B*1

        ssl2_loss += -tf.reduce_mean(tf.log(self.pos_score / self.ttl_score))   # 数字

        return ssl2_loss
    def cal_crloss(self, a_embedding, b_embedding, c_embedding):
        """
        cos(a, b) > cos(a, c)
        Args:
            a_embedding:
            b_embedding:
            c_embedding:

        Returns: loss
        """
        ab_inner = tf.reduce_sum(a_embedding * b_embedding, axis=1)
        ac_inner = tf.reduce_sum(a_embedding * c_embedding, axis=1)

        a_mod = tf.sqrt(tf.reduce_sum(tf.square(a_embedding), axis=1))
        b_mod = tf.sqrt(tf.reduce_sum(tf.square(b_embedding), axis=1))
        c_mod = tf.sqrt(tf.reduce_sum(tf.square(c_embedding), axis=1))

        ab_cos = ab_inner / a_mod / b_mod
        ac_cos = ac_inner / a_mod / c_mod

        res_loss = tf.reduce_mean(tf.square(1 - tf.nn.sigmoid(ab_cos - ac_cos)))

        return res_loss

    def create_model_str(self):
        log_dir = '/'+self.alg_type+'/layers_'+str(self.n_layers)+'/dim_'+str(self.emb_dim)
        log_dir+='/'+args.dataset+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir


    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

        print('using xavier initialization')
        self.weight_size_list = [self.emb_dim] + self.weight_size

        all_weights['W_mlp_sigma1'] = tf.Variable(
            initializer([self.emb_dim, self.emb_dim]), name='W_mlp_sigma1')
        all_weights['b_mlp_sigma1'] = tf.Variable(
            initializer([1, self.emb_dim]), name='b_mlp_sigma1')
        all_weights['W_mlp_sigma2'] = tf.Variable(
            initializer([self.emb_dim, self.emb_dim]), name='W_mlp_sigma2')
        all_weights['b_mlp_sigma2'] = tf.Variable(
            initializer([1, self.emb_dim]), name='b_mlp_sigma2')

        all_weights['Liner_W1'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['Liner_W2'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['Liner_W3'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['Liner_W4'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['Liner_W5'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['Liner_W6'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))

        all_weights['projection1'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['projection2'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['projection3'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['projection4'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['projection5'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))
        all_weights['projection6'] = tf.Variable(initializer([self.emb_dim, self.emb_dim]))

        all_weights['bias1'] = tf.Variable(initializer([1, self.emb_dim]))
        all_weights['bias2'] = tf.Variable(initializer([1, self.emb_dim]))
        all_weights['bias3'] = tf.Variable(initializer([1, self.emb_dim]))
        all_weights['bias4'] = tf.Variable(initializer([1, self.emb_dim]))
        all_weights['bias5'] = tf.Variable(initializer([1, self.emb_dim]))
        all_weights['bias6'] = tf.Variable(initializer([1, self.emb_dim]))

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_gc_pv_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_pv_%d' % k)
            all_weights['b_gc_pv_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_pv_%d' % k)

            all_weights['W_gc_cart_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_cart_%d' % k)
            all_weights['b_gc_cart_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_cart_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings=side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        return all_embeddings


    def _create_pv_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.pv_adj)
        else:
            A_fold_hat = self._split_A_hat(self.pv_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings=side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        return all_embeddings


    def _create_cart_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.cart_adj)
        else:
            A_fold_hat = self._split_A_hat(self.cart_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings=side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        return all_embeddings


    def _create_fav_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.fav_adj)
        else:
            A_fold_hat = self._split_A_hat(self.fav_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings=side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        return all_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss


    def create_kl_loss(self):
        KL_u = (0.5/self.n_users) * ( tf.reduce_mean(
            tf.reduce_sum( (-self.logsig_u + tf.exp(self.logsig_u) + self.mu_u ** 2 - 1), axis=1)) )
        KL_i = (0.5/self.n_items) * ( tf.reduce_mean(
            tf.reduce_sum( (-self.logsig_i + tf.exp(self.logsig_i) + self.mu_i ** 2 - 1), axis=1)) )
        KL = KL_u + KL_i
        return KL


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    pre_adj,pre_adj_pv,pre_adj_cart , pre_adj_fav= data_generator.get_adj_mat()


    config['norm_adj']=pre_adj
    config['pv_adj'] = pre_adj_pv
    config['cart_adj'] = pre_adj_cart
    config['fav_adj'] = pre_adj_fav

    config['pv_mat'] = data_generator.R_pv
    config['cart_mat'] = data_generator.R_cart
    config['fav_mat'] = data_generator.R_fav
    config['buy_mat'] = data_generator.R


    train_items = copy.deepcopy(data_generator.train_items)
    pv_set = copy.deepcopy(data_generator.pv_set)
    cart_set = copy.deepcopy(data_generator.cart_set)
    fav_set = copy.deepcopy(data_generator.fav_set)


    print('use the pre adjcency matrix')

    t0 = time()


    pretrain_data = None

    model = LightGCN(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6) # 改变这个百分比即可
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')



    tensorboard_model_path = './tensorboard'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path+model.log_dir +'/run_' + str(run_time)):
            run_time += 1
        else:
            break
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False



    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss, kl_loss = 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        loss_test,mf_loss_test,emb_loss_test,reg_loss_test, kl_loss_test  =0.,0.,0.,0., 0.
        # -------------  train   ----------------------
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_kl_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss, model.kl_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items,
                                          model.is_training_ph: 1
                                          })
            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            emb_loss += batch_emb_loss/n_batch
            kl_loss += batch_kl_loss/n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()


        if (epoch + 1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss,  kl_loss)
                print(perf_str)
                model.logger.info(perf_str)
            continue



        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        t3 = time()
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f ], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, kl_loss_test, ret['recall'][0], ret['recall'][1],
                        ret['precision'][0], ret['precision'][1], ret['hit_ratio'][0], ret['hit_ratio'][1],
                        ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)
            model.logger.info(perf_str)


        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][1], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)


        if should_stop == True:
            break


        if ret['ndcg'][1] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)


    best_ndcg_1 = max(ndcgs[:, 1])
    idx = list(ndcgs[:, 1]).index(best_ndcg_1)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    model.logger.info(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'embed_size=%d, lr=%.4f,tradeOff_ssl=%.4f,ssl_temp=%.4f,tradeOff_cr=%.4f,tradeOff=%.4f, layer_size=%s, node_dropout_flag=%s, node_dropout=%s, mess_dropout=%s, regs=%s,tst_file=%s adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.tradeOff_ssl ,args.ssl_temp, args.tradeOff_cr, args.tradeOff ,args.layer_size,args.node_dropout_flag, args.node_dropout, args.mess_dropout, args.regs,args.tst_file,
           args.adj_type, final_perf))
    f.close()
