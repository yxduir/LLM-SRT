import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


        
class EncoderProjectorQFormerPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers
        

        self.query_len = int(config.get("query_len", 150))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)


        self.linear1 = nn.Linear(configuration.hidden_size, 2560)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2560, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)


    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)

        # print("encoder_output",query.shape)

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )

        

        q_out = query_output.last_hidden_state  # (B, T, D)
        B, T, D = q_out.size()

        # print("query_output",q_out.shape)

        # ---- 平均池化下采样 ----
        num_frames_to_discard = T % self.k
        if num_frames_to_discard > 0:
            q_out = q_out[:, :-num_frames_to_discard, :]
            T = q_out.size(1)

        q_out = q_out.view(B, T // self.k, self.k, D).mean(2)  # (B, T/k, D)

        # print("q_out",q_out.shape)

        # 两层 MLP
        q_out = self.linear1(q_out)
        q_out = self.relu(q_out)
        q_out = self.linear2(q_out)

        query_proj = self.norm(q_out)

        # print("query_proj",query_proj.shape)

        return query_proj
    

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers
        

        self.query_len = int(config.get("query_len", 80))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        # (encoder维度)1280->2048(llm维度)   3B
        # 1280->5120    32B
        if self.llm_dim <= 1536:
            self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)
        elif self.llm_dim <= 2560:
            self.linear1 = nn.Linear(configuration.hidden_size, 1536)  # 从 768 -> 2560
            self.relu = nn.ReLU()  # 激活函数
            self.linear2 = nn.Linear(1536, self.llm_dim)  # 从 2560 -> 5120
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)  # 最终归一化
        else:
            self.linear1 = nn.Linear(configuration.hidden_size, 2560)  # 从 768 -> 2560
            self.relu = nn.ReLU()  # 激活函数
            self.linear2 = nn.Linear(2560, self.llm_dim)  # 从 2560 -> 5120
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)  # 最终归一化
        
       

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        if self.llm_dim <= 1536:
            query_proj = self.norm(self.linear(query_output.last_hidden_state))
        else:
            x = self.linear1(query_output.last_hidden_state)  # 从 1280 -> 2560
            x = self.relu(x)  # 激活
            x = self.linear2(x)  # 从 2560 -> 5120
            query_proj = self.norm(x)  # LayerNorm 归一化

        
        return query_proj