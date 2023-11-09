import copy
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import matplotlib

matplotlib.use('Agg')
import torch

class CascadeSemanticAlignmentNetwork(nn.Module):

    def __init__(self, n_layers=3, d_in=1024, d_model=512, n_heads=8, d_ff=2048, dropout=0.1,
                 activation="relu", **kwargs,):
        super().__init__()

        # object semantic features aggregation
        self.object_semantic_agg_k = nn.Sequential(nn.Linear(d_in, d_in),
                                          nn.ReLU(),
                                          nn.Linear(d_in, 150))
        self.object_semantic_agg_v = nn.Linear(d_in, d_model)

        ## relation semantic features aggregation
        self.relation_semantic_agg_k = nn.Sequential(nn.Linear(d_in, d_in),
                                          nn.ReLU(),
                                          nn.Linear(d_in, 150))
        self.relation_semantic_agg_v = nn.Linear(d_in, d_model)

        self.object_encoder_layer0 = TransformerEncoderLayer_SelfAtten(d_model, n_heads, d_ff,
                                                dropout, activation, )
        self.object_encoder_layer1 = TransformerEncoderLayer_SelfAtten(d_model, n_heads, d_ff,
                                                dropout, activation, )
        self.relation_encoder_layer2 = TransformerEncoderLayer_SelfAtten(d_model, n_heads, d_ff,
                                                dropout, activation, )

        self.object_semantic_encoder_layer0 = TransformerEncoderLayer_CrossAtten(d_model, n_heads, d_ff,
                                                dropout, activation, )
        self.object_semantic_encoder_layer1 = TransformerEncoderLayer_CrossAtten(d_model, n_heads, d_ff,
                                                dropout, activation, )
        self.relation_semantic_encoder_layer2 = TransformerEncoderLayer_CrossAtten(d_model, n_heads, d_ff,
                                                dropout, activation, )

        self.response_gate0 = nn.Sequential(nn.Linear(d_model*2, 1, bias=False), nn.Sigmoid())
        self.out_proj0 = nn.Linear(d_model, d_model)

        self.response_gate1 = nn.Sequential(nn.Linear(d_model*2, 1, bias=False), nn.Sigmoid())
        self.out_proj1 = nn.Linear(d_model, d_model)

        self.response_gate2 = nn.Sequential(nn.Linear(d_model*2, 1, bias=False), nn.Sigmoid())
        self.out_proj2 = nn.Linear(d_model, d_model)

        self._reset_parameters()
        #object regionclip text embedding
        self.object_weight = torch.load('./coco_caption/coco_nouns_emb.pth')
        #relation regionclip text embedding
        self.relation_weight = torch.load('./coco_caption/coco_verbs_emb.pth')

        self.d_model = d_model
        self.n_heads = n_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vis_inputs):
        src = vis_inputs['reg_feat']

        # object semantic features aggregation
        object_semantic_k = self.object_semantic_agg_k(self.object_weight.to(src.device)).softmax(-2) #[1024,150]
        object_semantic_v = self.object_semantic_agg_v(self.object_weight.to(src.device)) #[1024,512]
        agg_nouns_features = torch.matmul(object_semantic_k.permute(1,0), object_semantic_v) #[150,512]
        nouns = agg_nouns_features.unsqueeze(0).repeat(src.shape[0], 1, 1)  # [b, n ,d]

        # verb aggregation
        verb_k_proj = self.relation_semantic_agg_k(self.relation_weight.to(src.device)).softmax(-2) #[1024,150]
        verb_v_proj = self.relation_semantic_agg_v(self.relation_weight.to(src.device)) #[1024,512]
        agg_verbs_features = torch.matmul(verb_k_proj.permute(1,0), verb_v_proj) #[150,512]
        verbs = agg_verbs_features.unsqueeze(0).repeat(src.shape[0], 1, 1)  # [b, n ,d]

        # textual-visual alignment layer0 (object)
        visual_features = src.permute(1, 0, 2) #HW xb xC
        element_router_attn = (nouns @ visual_features.permute(1, 2, 0)).contiguous()
        pool_weights = torch.softmax(element_router_attn / 0.1, -1)
        pooled_features = (element_router_attn * pool_weights).sum(-1)
        element_router_attn = torch.softmax(pooled_features/ 0.05, -1).unsqueeze(-1)
        evolution_word_embs0 = element_router_attn * nouns
        evolution_nouns_word_embs0 = evolution_word_embs0.permute(1, 0, 2)

        # Cross-Modal Deep Fusion layer0
        # Self-modal Multi-Head Attention
        context_region_layer0 = self.object_encoder_layer0(visual_features) # [N,B,Dim]
        # Cross-modal Multi-Head Attention
        context_region_nouns_layer0 = self.object_semantic_encoder_layer0(context_region_layer0, evolution_nouns_word_embs0)
        # adaptive fusion
        response_gate = self.response_gate0(torch.cat([context_region_layer0, context_region_nouns_layer0], dim=2))
        x = context_region_layer0 * (1 - response_gate) + context_region_nouns_layer0 * response_gate
        context_region_nouns_layer0 = self.out_proj0(x)

        # textual-visual alignment layer1 (object second layers)
        element_router_attn = (evolution_word_embs0 @ context_region_nouns_layer0.permute(1, 2,0)).contiguous()
        pool_weights = torch.softmax(element_router_attn / 0.1, -1)  #
        pooled_features = (element_router_attn * pool_weights).sum(-1)
        element_router_attn = torch.softmax(pooled_features/ 0.05, -1).unsqueeze(-1)
        evolution_word_embs1 = element_router_attn * evolution_word_embs0
        evolution_nouns_word_embs1 = evolution_word_embs1.permute(1, 0, 2)

        # Cross-Modal Deep Fusion layer1 (object second layers)
        context_region_layer1 = self.object_encoder_layer1(context_region_nouns_layer0)
        context_region_nouns_layer1 = self.object_semantic_encoder_layer1(context_region_layer1, evolution_nouns_word_embs1)
        response_gate = self.response_gate1(torch.cat([context_region_layer1, context_region_nouns_layer1], dim=2))
        x = context_region_layer1 * (1 - response_gate) + context_region_nouns_layer1 * response_gate
        context_region_nouns_layer1 = self.out_proj1(x)
        object_grounded_features = context_region_nouns_layer1.transpose(0, 1)

        # textual-visual alignment layer2 (relation first layer)
        element_router_attn = (verbs @ context_region_nouns_layer1.permute(1, 2, 0)).contiguous()
        pool_weights = torch.softmax(element_router_attn / 0.1, -1)
        pooled_features = (element_router_attn * pool_weights).sum(-1)
        element_router_attn = torch.softmax(pooled_features/ 0.05, -1).unsqueeze(-1)
        evolution_word_embs2 = element_router_attn * verbs
        evolution_relation_word_embs2 = evolution_word_embs2.permute(1, 0, 2)

        #Cross-Modal Deep Fusion (relation first layer)
        context_region_layer2 = self.relation_encoder_layer2(context_region_nouns_layer1)
        context_region_relation_layer2 = self.relation_semantic_encoder_layer2(context_region_layer2, evolution_relation_word_embs2)
        response_gate = self.response_gate2(torch.cat([context_region_layer2, context_region_relation_layer2], dim=2)) # dim
        x = context_region_layer2 * (1 - response_gate) + context_region_relation_layer2 * response_gate
        relation_grounded_features = self.out_proj2(x).transpose(0, 1)

        rel_mask = (torch.sum(relation_grounded_features, dim=-1) == 1)
        rel_mask = rel_mask.unsqueeze(1).unsqueeze(1)

        return object_grounded_features, relation_grounded_features, rel_mask


class TransformerEncoderLayer_SelfAtten(nn.Module):
    """self attention layer"""
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.activation = _get_activation_fn(activation)

        # Self-Atten
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, entity):

        #self-attention
        tgt2 = self.self_attn(query=entity, key=entity, value=entity)[0] #不取 attention map
        tgt = entity + self.dropout0(tgt2)
        tgt = self.norm0(tgt)
        context_entity = self.forward_ffn(tgt)

        return context_entity

class TransformerEncoderLayer_CrossAtten(nn.Module):
    """cross attention layer"""
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.activation = _get_activation_fn(activation)

        # cross attention
        self.cross_text_entity = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, context_entity, semantic):

        # cross-attention
        tgt2 = self.cross_text_entity(query=context_entity, key=semantic, value=semantic)[0]

        tgt = context_entity + self.dropout0(tgt2)
        tgt = self.norm0(tgt)
        context_region_nouns = self.forward_ffn(tgt)

        return context_region_nouns

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
