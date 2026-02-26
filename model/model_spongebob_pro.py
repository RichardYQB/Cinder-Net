import math
from mpmath import residual
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .config import SpongeBobConfig#导入当前目录的参数配置

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int , eps: float=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def _norm(self,x):
        return x*torch.rqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    def forward(self,x):
        output=self._norm(x.float()).type_as(x)
        return output*self.weight

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用 RoPE
    Args:
        position_ids: (batch, seq_len) 用于推理时指定位置
    """
    # 1. 处理 cos/sin 的切片 (支持推理/KV Cache)
    # 如果传入了 position_ids，则根据 id 选取对应的 cos/sin
    if position_ids is not None:
        # cos: (end, dim) -> (batch, seq_len, dim)
        cos = cos[position_ids]
        sin = sin[position_ids]
        
        # 此时 cos/sin 已经是 (batch, seq_len, dim)，我们需要广播到 head 维度
        # q: (batch, seq, heads, dim)
        # 这种情况下通常 unsqueeze_dim=2 (heads维度)
        cos = cos.unsqueeze(2) # (batch, seq, 1, dim)
        sin = sin.unsqueeze(2)
    else:
        # 兼容旧逻辑：假设 seq_len 从 0 开始且连续 (训练时常用)
        # 截取当前序列长度
        seq_len = q.shape[1]
        cos = cos[:seq_len].unsqueeze(unsqueeze_dim) # (seq, 1, dim)
        sin = sin[:seq_len].unsqueeze(unsqueeze_dim)

    # 2. 修正后的 rotate_half (LLaMA 风格)
    # 配合 precompute 的 cat([cos, cos])，这里必须将 tensor 切分为前后两半
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1) # 将最后一维切分成两半
        return torch.cat((-x2, x1), dim=-1)

    # 3. 计算
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6):
    """
    预计算 RoPE (Rotary Position Embedding) 的 cos 和 sin 频率
    
    Args:
        dim: 注意力头的维度 (head_dim)
        end: 最大序列长度
        rope_base: RoPE 的基础频率，默认 1e6
    
    Returns:
        freqs_cos: cos 频率张量 (end, dim)
        freqs_sin: sin 频率张量 (end, dim)
    """
    # 计算频率：θ_i = base^(-2i/d), i ∈ [0, d/2)，生成 dim//2 个频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 计算每个位置的频率：pos * θ_i，形状 (end, dim//2)
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs).float()
    
    # 计算 cos 和 sin，并复制一次以匹配 head_dim（用于两两分组）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # (end, dim)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # (end, dim)
    
    return freqs_cos, freqs_sin

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对 KV 进行重复以匹配 Query 的头数（用于 Grouped Query Attention）
    等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)，但更高效
    
    Args:
        x: KV 张量 (batch, seq_len, num_kv_heads, head_dim)->(batch, seq_len, num_kv_heads,n_rep, head_dim)  num_heads=num_kv_heads*n_rep
        n_rep: 重复次数 (num_heads // num_kv_heads)
    
    Returns:
        重复后的张量 (batch, seq_len, num_heads, head_dim)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self,args: SpongeBobConfig):
        super().__init__
        self.num_key_value_heads=args.num_key_value_heads#每一组里包含3个query
        assert args.num_attention_heads%self.num_key_value_heads==0#query的个数必须是num_key_value_heads的整数倍
        self.num_heads=args.num_attention_heads#每个head包含的query的个数
        self.num_kv_heads=self.num_key_value_heads#KV头数
        self.n_rep=self.num_heads//self.num_kv_heads#每个KV头重复的次数
        self.head_dim=args.hidden_size//self.num_heads#每个head的维度
        #参数
        #q;512->512
        #k:512->128
        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,args.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)
        #Dropout
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout
    
    

    def forward(self,
        x:torch.Tensor,
        position_embeddings:Tuple[torch.Tensor,torch.Tensor],
        past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
        use_cache=False,
        attention_mask:Optional[torch.Tensor]=None):

        bsz,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        #将hidden_size拆成了num_heads*head_dim 和 num_kv_heads*head_dim
        xq=xq.view(bsz,seq_len,self.num_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.num_kv_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.num_kv_heads,self.head_dim)
        #应用RoPE位置编码
        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)#进去了一个q 和 K,出来一个q和K

        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)#cat是把两个向量粘在一起
            xv=torch.cat([past_key_value[1],xv],dim=1)#dim=1是在sequence的维度上进行粘粘
        past_kv=(xk,xv) if use_cache else None
        #(batch_size,seq_len,num_heads,head_dim)->(batch_size,num_heads,seq_len,head_dim)
        xq,xk,xv=(
            xq.transpose(1,2),
            repeat_kv(xk,self.n_rep).transpose(1,2),
            repeat_kv(xv,self.n_rep).transpose(1,2),
        )

        #attention 计算
        scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
        #应用causa_mask(上三角设为-inf(负无穷大))
        scores[:,:,:,-seq_len:]+=torch.triu(torch.full((seq_len,seq_len),float("-inf")),device=scores.device,diagonal=1)
        if attention_mask is not None:
            extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask=(1.0-extended_attention_mask)*-1e9
            scores+=extended_attention_mask

        scores=F.softmax(scores.float(),dim=-1).type_as(xq)
        scores=self.attn_dropout(scores)
        output=scores@xv
        #output:(batch_size,num_heads,seq_len,head_dim)->(batch_size,seq_len,num_heads*head_dim)
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)
        output=self.o_proj(output)
        output=self.resid_dropout(output)
        return output,past_kv

class FeedForward(nn.Module):
    """
    前馈神经网络（SwiGLU 激活函数）
    结构: Gate(x) * Up(x) -> Down
    """
    def __init__(self, config: SpongeBobConfig):
        super().__init__()
        # 计算中间层大小：默认为 hidden_size * 8/3，向上取整到 64 的倍数
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)#llama的实验得出8/3倍
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 向上取整到 64 的倍数
        
        self.gate_proj=nn.Linear(config.hidden_size,intermediate_size,bias=False)#768->2048
        self.down_proj=nn.Linear(intermediate_size,config.hidden_size,bias=False)
        self.up_proj=nn.Linear(config.hidden_size,intermediate_size,bias=False)

        self.dropout=nn.Dropout(config.dropout)
        self.act_fn=ACT2FN[config.hidden_act]#silu激活函数

    def foeward(self,x):
        #SwiGLU:  act(gate_proj(x))*up_proj(x)->down_proj(act(gate_proj(x))*up_proj(x))
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))

class SpongeBobBlock(nn.Module):
    """
    Transformer 块：Self-Attention + FeedForward
    采用 Pre-Norm 结构（Norm before attention/mlp）
    """
    def __init__(self, layer_id: int, config: SpongeBobConfig):
        super().__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=config.hidden_size//config.num_attention_heads
        self.self_attn=Attention(config)

        self.layer_id=layer_id
        #LayerNorm
        self.input_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)#Attention之前的norm
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)#Attention之后的norm
        self.mlp=FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播：Pre-Norm Transformer Block
        输入：hidden_states: (batch_size, seq_len, hidden_size)
        position_embeddings: (cos, sin)
        past_key_value: (key, value)
        use_cache: 是否使用cache
        attention_mask: 注意力掩码
        输出：hidden_states: (batch_size, seq_len, hidden_size)
        past_key_value: (key, value)
        
        结构：
            x = x + Attention(Norm(x))
            x = x + MLP(Norm(x))
        """
        #self-Attention层
        residual=hidden_states#先暂存输入
        hidden_states,present_key_value=self.self_attn(self.input_layernorm(hidden_states),position_embeddings,past_key_value,use_cache,attention_mask)
        hidden_states=hidden_states+residual#做残差
        #FFN
        residual=hidden_states#先暂存输入
        hidden_states=self.post_attention_layernorm(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=hidden_states+residual#做残差
        return hidden_states,present_key_value

class SpongeBobModel(nn.Module):
    """
    SpongeBob 模型主体（Decoder-only Transformer）
    """
    def __init__(self, config: SpongeBobConfig):
        super().__init__()
        self.config=config
        self.vocab_size,self.num_hidden_layers=config.vocab_size,config.num_hidden_layers

        #Token Embedding
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)

        self.layers=nn.ModuleList([SpongeBobBlock(i,config) for i in range(self.num_hidden_layers)])

        #最终的layerNorm
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        #预计算RoPE频率（注册为buffer，不参与训练，但会保存在模型中
        freqs_cos,freqs_sin=precompute_freqs_cis(dim=config.hidden_size//config.num_attention_heads,end=config.max_position_embeddings,rope_base=config.rope_theta)
        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,#它可以是 torch.Tensor 类型，也可以是 None
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)，1=有效位置，0=padding
            past_key_values: KV cache 列表，用于推理加速
            use_cache: 是否返回新的 KV cache
        
        Returns:
            hidden_states: 最后一层的隐藏状态 (batch, seq_len, hidden_size)
            presents: 新的 KV cache 列表
        """
        batch_size,seq_len=input_ids.shape
        if hasattr(past_key_values,'layers'):
            past_key_values=None
        past_key_values=past_key_values or [None] * self.layers#如果past_key_values是None，则初始化为None
        #计算起始位置（用于RoPE）
        start_pos=past_key_values[0][0].shape[1] if past_key_values is not None else 0
        #Token Embedding+dropout
        hidden_states=self.dropout(self.embed_tokens(input_ids))
        #获取当前序列的位置编码（从start.pos开始）
        position_embeddings=(self.freqs_cos[start_pos:start_pos+seq_len],
                            self.freqs_sin[start_pos:start_pos+seq_len])

        #逐层前向传播
        presents=[]
        for layer_idx,(layer,past_key_value) in enumerate(zip(self.layers,past_key_values)):
            hidden_states,present_key_value=layer(hidden_states,position_embeddings,past_key_value,use_cache,attention_mask)
            presents.append(present_key_value)
        hidden_states=self.norm(hidden_states)
        return hidden_states,presents


class SpongeBobForCausalLM(PreTrainedModel,GenerationMixin):
    """
    SpongeBob 因果语言模型（Causal Language Model）
    在SpongeBob基础上添加 Language Model Head
    """
    config_class=SpongeBobConfig

    def __init__(self, config: SpongeBobConfig=None):
        self.config=config or SpongeBobConfig()
        super().__init__(self.config)
     
        #Transformer主体
        self.model=SpongeBobModel(self.config)
        #Language Model Head(与embed_tokens权重共享)
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        self.model.embed_tokens.weight=self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播（用于训练和推理）
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)
            labels: 标签 (batch, seq_len)，用于计算 loss
            past_key_values: KV cache
            use_cache: 是否返回 KV cache
            logits_to_keep: 保留最后多少个 token 的 logits（节省内存）
        
        Returns:
            CausalLMOutputWithPast: 包含 loss, logits, past_key_values, hidden_states
        """
        # Transformer 前向传播
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else slice(None,logits_to_keep)
        logits=self.lm_head(hidden_states[:,slice_indices,:])
        #计算交叉熵损失
        loss=None
        if labels is not None:
            #标准的自回归模型loss计算：
            #预测token[i+1],使用token[i]的信息
            #shift_logits:[0,1,...,n-2]位置的预测
            #shift_labels:[1,2,...,n-1]位置的标签
            shift_logits=logits[...,:-1,:].contiguous()
            shift_labels=labels[...,1:].contiguous()
            loss=F.cross_entropy(
                shift_logits.view(-1,shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,#忽略padding和mask的位置,padding位置的label为-100
            )
            output=CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
            )#记录器可以用来记录训练过程中的各种信息，比如loss,logits,past_key_values,hidden_states等
        return output
        
        
        