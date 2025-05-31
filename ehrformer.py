import torch.nn as nn
import torch
from einops import rearrange
from transformers import BertConfig, GPT2Model, BertModel
from torch.autograd import Function
import torch.nn.functional as F


class MultiTaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['transformer'].hidden_size
        self.n_cls = [1 for _ in range(len(config['current_label']["reg_label_cols"]))]

        self.cls_fcs = nn.ModuleList(nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n)
        ) for n in self.n_cls)

    def forward(self, x, B):
        results = []
        for i, cls_fc in enumerate(self.cls_fcs):
            y_cls = cls_fc(x)
            y_cls = rearrange(y_cls, '(b l) d -> b l d', b=B)
            results.append(y_cls)
        
        return results


class EHREmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer(
            "token_type_ids", torch.arange(config['n_category_feats']+config['n_float_feats']+1), persistent=False
        )
        self.register_buffer(
            "one", torch.ones(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "zero", torch.zeros(1, dtype=torch.long), persistent=False
        )

        self.bert = BertModel(BertConfig(
            vocab_size=config['n_category_values']+config['n_float_values']+2,  # 1 for padding, 0 for CLS
            hidden_size=config['transformer'].hidden_size,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=config['transformer'].hidden_size * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=config['n_category_feats']+config['n_float_feats']+1,  # 0 for CLS
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            position_embedding_type="none",
            use_cache=True,
            classifier_dropout=None
        ))

    def forward(self, cat_feats, float_feats):  # cat_feats: (b, nc, l), float_feats: (b, nf, l)

        B = cat_feats.shape[0]
        L = cat_feats.shape[2]
        cat_feats_mask = cat_feats == -1
        float_feats_mask = float_feats == -1
        attention_mask = torch.cat([self.one.unsqueeze(1).unsqueeze(0).expand(B, -1, L), ~cat_feats_mask, ~float_feats_mask], dim=1)
        attention_mask = rearrange(attention_mask, 'b n l -> (b l) n')

        cat_feats = cat_feats + 2
        cat_feats[cat_feats_mask] = 1
        float_feats = float_feats + 2 + self.config['n_category_values']
        float_feats[float_feats_mask] = 1
        input_ids = torch.cat([self.zero.unsqueeze(1).unsqueeze(0).expand(B, -1, L), cat_feats, float_feats], dim=1)
        input_ids = rearrange(input_ids, 'b n l -> (b l) n')

        BL = input_ids.shape[0]
        token_type_ids = self.token_type_ids.unsqueeze(0).expand(BL, -1)
        ft_emb = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).pooler_output
        ft_emb = rearrange(ft_emb, '(b l) d -> b l d', b=B)
        return ft_emb  # time_index: (b, l, d)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class EHRFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.ehr_embed = EHREmbedding(config)
        self.transformer = GPT2Model(config['transformer'])
        self.head = MultiTaskHead(config)
        # Adversarial head for medication type
        self.medication_adv_head = nn.Sequential(
            nn.Linear(config['transformer'].hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, config['n_medication_types'])
        )

    def forward(self, cat_feats, float_feats, valid_mask, time_index, medication_type=None, grl_lambda=1.0):
        B = cat_feats.shape[0]
        ft_emb = self.ehr_embed(cat_feats, float_feats)
        y = self.transformer(
            inputs_embeds=ft_emb,
            position_ids=time_index,
            attention_mask=valid_mask
        ).last_hidden_state
        y = rearrange(y, 'b l d -> (b l) d')
        y_cls = self.head(y, B)

        adv_loss = None
        if medication_type is not None:
            # Use mean pooling over sequence
            patient_summary_representation = y.mean(dim=1)
            grl_out = grad_reverse(patient_summary_representation, grl_lambda)
            medication_logits = self.medication_adv_head(grl_out)
            adv_loss = F.cross_entropy(medication_logits, medication_type)
        return y_cls, adv_loss
