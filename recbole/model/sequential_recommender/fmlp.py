r"""
MLP4Rec v1.7

无线性output层 一维卷积加速

"""
import os

import numpy as np
import torch
from torch import nn
from functools import partial
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
import argparse
from modules import Encoder, LayerNorm


class FMLP(SequentialRecommender):
    r"""
    FDSA is similar with the GRU4RecF implemented in RecBole, which uses two different Transformer encoders to
    encode items and features respectively and concatenates the two subparts' outputs as the final output.

    """

    def __init__(self, config, dataset, seq_len = 50):
        super(FMLP, self).__init__(config, dataset)

        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", default="./data/", type=str)
        parser.add_argument("--output_dir", default="output/", type=str)
        parser.add_argument("--data_name", default="Beauty", type=str)
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--load_model", default=None, type=str)

        # model args
        parser.add_argument("--model_name", default="FMLPRec", type=str)
        parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
        parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
        parser.add_argument("--num_attention_heads", default=2, type=int)
        parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
        parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
        parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
        parser.add_argument("--initializer_range", default=0.02, type=float)
        parser.add_argument("--max_seq_length", default=50, type=int)
        parser.add_argument("--no_filters", action="store_true",
                            help="if no filters, filter layers transform to self-attention")

        # train args
        parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
        parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
        parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
        parser.add_argument("--no_cuda", action="store_true")
        parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
        parser.add_argument("--full_sort", action="store_true")
        parser.add_argument("--patience", default=10, type=int,
                            help="how long to wait after last time validation loss improved")

        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
        parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
        parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
        parser.add_argument("--variance", default=5, type=float)

        args = parser.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

        args.item_size = 12102



        # load parameters info
        self.n_layers = args.num_hidden_layers
        self.n_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.hidden_act = args.hidden_act
        self.selected_features = config['selected_features']
        self.device = config['device']

        self.initializer_range = args.initializer_range
        self.loss_type = config['loss_type']

        
        # if self.loss_type == 'BPR':
        #     self.loss_fct = BPRLoss()
        # elif self.loss_type == 'CE':
        #     # self.loss_fct = nn.CrossEntropyLoss()
        #     self.loss_fct = self.calculate_loss
        # else:
        #     raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)

        # parameters initialization
        self.apply(self._init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, item_seq_len):
        # item_seq shape: torch.Size([256, 50]), item_seq_len shape: torch.Size([256])
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]
        return sequence_output




    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        item_num = seq_output.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]

        pos_items = pos_items.reshape((-1,1))
        pos_items = torch.cat((item_seq.clone()[:, 1:], pos_items), 1)
        neg_items = torch.randint(self.args.item_size, (item_num, 50)).cuda()
        # print(pos_items.shape)
        # print(neg_items.shape)
        # for i in range(256):
        #     for j in range(50):
        #         if neg_items[i,j]==pos_items[i,j]:
        #             neg_items[i,j]+=1
        #             if neg_items[i,j]>=self.args.item_size:
        #                 neg_items[i,j]=0
        if self.loss_type == 'CE':
            # neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embeddings(pos_items)
            neg_emb = self.item_embeddings(neg_items)
            seq_emb = seq_output
            # print(pos_emb.shape)
            pos_logits = torch.sum(pos_emb * seq_emb, -1)  # [batch*seq_len]
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            # istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
            loss = torch.mean(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
            )  # / torch.sum(istarget)
        else:
            print("loss error")

        return loss

    def loss_fct(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        item_num = seq_output.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]

        pos_items = pos_items.reshape((-1,1))
        pos_items = torch.cat((item_seq.clone()[:, 1:], pos_items), 1)
        neg_items = torch.randint(self.args.item_size, (item_num, 50)).cuda()
        # print(pos_items.shape)
        # print(neg_items.shape)
        # for i in range(256):
        #     for j in range(50):
        #         if neg_items[i,j]==pos_items[i,j]:
        #             neg_items[i,j]+=1
        #             if neg_items[i,j]>=self.args.item_size:
        #                 neg_items[i,j]=0
        if self.loss_type == 'CE':
            # neg_items = interaction[self.NEG_ITEM_ID]
            pos_emb = self.item_embeddings(pos_items)
            neg_emb = self.item_embeddings(neg_items)
            seq_emb = seq_output
            # print(pos_emb.shape)
            pos_logits = torch.sum(pos_emb * seq_emb, -1)  # [batch*seq_len]
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            # istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
            loss = torch.mean(
                - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
            )  # / torch.sum(istarget)
        else:
            print("loss error")

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)[:,-1,:]
        test_item_emb = self.item_embeddings(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)[:,-1,:]
        test_items_emb = self.item_embeddings.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
