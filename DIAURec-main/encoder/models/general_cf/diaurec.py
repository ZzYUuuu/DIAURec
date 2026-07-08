import pickle
import torch as t
import torch
from torch import nn
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, alignment, uniformity, coarse_match, fine_match
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
import torch.nn.functional as F
from hyperspherical_vae.distributions import VonMisesFisher
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DIAURec(BaseModel):
    def __init__(self, data_handler):
        super(DIAURec, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.intent_size = self.hyper_config['intent_size']
        self.user_text_intent = torch.nn.Embedding(num_embeddings=self.embedding_size,
                                              embedding_dim=self.intent_size)
        self.item_text_intent = torch.nn.Embedding(num_embeddings=self.embedding_size,
                                              embedding_dim=self.intent_size)
        self.user_id_intent = torch.nn.Embedding(num_embeddings=self.embedding_size,
                                              embedding_dim=self.intent_size)
        self.item_id_intent = torch.nn.Embedding(num_embeddings=self.embedding_size,
                                              embedding_dim=self.intent_size)
        nn.init.xavier_uniform_(self.user_text_intent.weight, gain=1.)
        nn.init.xavier_uniform_(self.item_text_intent.weight, gain=1.)
        nn.init.xavier_uniform_(self.user_id_intent.weight, gain=1.)
        nn.init.xavier_uniform_(self.item_id_intent.weight, gain=1.)
        
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False
        self.k = self.hyper_config['k']
        self.sigma = self.hyper_config['sigma']
        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.ssl_temperature = self.hyper_config['ssl_temperature']
        
        self.ssl_lambda1 = self.hyper_config['ssl_lambda1']
        self.ssl_lambda2 = self.hyper_config['ssl_lambda2']
        self.ssl_lambda3 = self.hyper_config['ssl_lambda3']
        self.ssl_lambda4 = self.hyper_config['ssl_lambda4']
        self.gamma = self.hyper_config['gamma']
        
        # semantic-embeddings
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.mlp1 = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self.fc_mu = torch.nn.Linear(self.embedding_size, self.embedding_size)

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp1:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.mlp2:
            if isinstance(m, nn.Linear):
                init(m.weight)
    
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
     
    def infonce_loss(self, embedding1, embedding2, temperature):
        embedding1 = torch.nn.functional.normalize(embedding1)
        embedding2 = torch.nn.functional.normalize(embedding2)

        pos_score = (embedding1 * embedding2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        total_score = torch.matmul(embedding1, embedding2.transpose(0, 1))
        total_score = torch.exp(total_score / temperature).sum(dim=1)

        cl_loss = -torch.log(pos_score / total_score + 10e-6)
        
        return torch.mean(cl_loss)
       
    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
         

        embeds = t.cat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        
        usrprf_embeds = self.mlp1(self.usrprf_embeds)
        itmprf_embeds = self.mlp2(self.itmprf_embeds)
        #gcn
        intent_embeds = t.cat([usrprf_embeds, itmprf_embeds], axis=0)
#         # w/o gcn
        intent_embeds_list = [intent_embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            intent_embeds = self._propagate(adj, intent_embeds)
            intent_embeds_list.append(intent_embeds)
        intent_embeds = sum(intent_embeds_list)
        usrprf_embeds, itmprf_embeds = torch.split(intent_embeds, [self.user_num, self.item_num])
        user_embeds3, item_embeds3 = torch.split(self.final_embeds, [self.user_num, self.item_num])
        user_text_intent = torch.softmax(usrprf_embeds @ self.user_text_intent.weight, dim=1) @ self.user_text_intent.weight.T  # [B, dim]
        item_text_intent = torch.softmax(itmprf_embeds @ self.item_text_intent.weight, dim=1) @ self.item_text_intent.weight.T  # [B, dim]

        mu = F.normalize(self.fc_mu(user_embeds3), dim=-1)  
        B = mu.size(0)
        kappa = torch.full((B, 1), self.k, device=mu.device, dtype=mu.dtype) 
        vmf = VonMisesFisher(mu, kappa)     
        user_intent_sample = vmf.rsample()   
        logits = user_intent_sample @ self.user_id_intent.weight
        attn_weights = torch.softmax(logits, dim=-1) 
        user_id_intent = torch.matmul(attn_weights, self.user_id_intent.weight.T) 
        
        
        mu = F.normalize(self.fc_mu(item_embeds3), dim=-1)  # [B, d]

        B = mu.size(0)
        kappa = torch.full((B, 1), self.k, device=mu.device, dtype=mu.dtype) 

        vmf = VonMisesFisher(mu, kappa)     
        item_intent_sample = vmf.rsample()   

        logits = item_intent_sample @ self.item_id_intent.weight 
        attn_weights = torch.softmax(logits, dim=-1)  
        item_id_intent = torch.matmul(attn_weights, self.item_id_intent.weight.T)  

        gnn_embeddings = torch.cat([user_embeds3, item_embeds3], dim=0)
        intent_text_embeddings = torch.cat([user_text_intent, item_text_intent], dim=0)
        intent_id_embeddings = torch.cat([user_id_intent, item_id_intent], dim=0)
        
        noise = torch.randn_like(gnn_embeddings)
        sigma = self.sigma
       
        final_embeddings = gnn_embeddings + intent_text_embeddings * noise * sigma + intent_id_embeddings * noise * sigma
        
        user_embeds3, item_embeds3 = torch.split(final_embeddings, [self.user_num, self.item_num])
        user_text_embeds, item_text_embeds = torch.split(intent_text_embeddings, [self.user_num, self.item_num])
        user_id_embeds, item_id_embeds = torch.split(intent_id_embeddings, [self.user_num, self.item_num])
        test_user, test_item = torch.split(gnn_embeddings, [self.user_num, self.item_num])
        
        user_base_embeds, item_base_embeds = torch.split(gnn_embeddings, [self.user_num, self.item_num])
        return user_embeds3, item_embeds3, user_text_embeds, item_text_embeds, user_id_embeds, item_id_embeds, user_base_embeds, item_base_embeds, usrprf_embeds, itmprf_embeds, embeds_list
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
      
        self.is_training = True
        ancs, poss, negs = batch_data
        

        #intent
        user_embeds3, item_embeds3, user_text_embeds, item_text_embeds, user_id_embeds, item_id_embeds, user_base_embeds, item_base_embeds, usrprf_embeds, itmprf_embeds, embeds_list3 = self.forward(self.adj, self.keep_rate)

        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        user_text_embed, item_text_embed, neg_text_embed = self._pick_embeds(user_text_embeds, item_text_embeds, batch_data)
        user_base_embed, item_base_embed, neg_base_embed = self._pick_embeds(user_base_embeds, item_base_embeds, batch_data)

        align_loss = alignment(anc_embeds3, pos_embeds3) 
        uniform_loss = uniformity(anc_embeds3)  

        d1, d2 = anc_embeds3.size(1), pos_embeds3.size(1)
        W1, W2 = torch.nn.Parameter(torch.eye(d1,device=anc_embeds3.device)), torch.nn.Parameter(torch.eye(d2,device=pos_embeds3.device))
        Coarse_loss = coarse_match(anc_embeds3, user_text_embed, W=W1, use_W=False) + coarse_match(pos_embeds3, item_text_embed, W=W2, use_W=False) 

        Fine_loss = fine_match(user_embeds3, user_text_embeds, ancs) + fine_match(item_embeds3, item_text_embeds, poss)

        Interaction_loss =    self.infonce_loss(anc_embeds3, user_base_embed, self.ssl_temperature) + \
                        self.infonce_loss(pos_embeds3, item_base_embed, self.ssl_temperature)
        Intra_space_loss =    self.infonce_loss(anc_embeds3, pos_embeds3, self.ssl_temperature) + \
                        self.infonce_loss(user_base_embed, item_base_embed, self.ssl_temperature)
        
        ssl_loss =  Coarse_loss * self.ssl_lambda1 + Fine_loss * self.ssl_lambda2 + Intra_space_loss * self.ssl_lambda3 + Interaction_loss * self.ssl_lambda4
        
        reg_loss = self.reg_weight * reg_params(self)

        loss = align_loss + self.gamma * uniform_loss + reg_loss + ssl_loss 

        losses = {'align_loss': align_loss, 'uniform_loss': uniform_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _, _, _, _, _, _, embeds_list = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
