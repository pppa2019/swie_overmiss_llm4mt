from typing import Optional, Tuple, Union
import torch
import warnings
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.bloom import BloomModel,BloomForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import BLOOM_INPUTS_DOCSTRING, BloomAttention, BloomBlock, BloomMLP, BloomPreTrainedModel, _expand_mask, _make_causal_mask, build_alibi_tensor
from transformers.utils import logging
from transformers.utils.doc import add_start_docstrings_to_model_forward

def token_ids2seg_ids(token_ids, seg_id):
        seg_flag = (token_ids==seg_id)
        return torch.cumsum(seg_flag, dim=-1)
logger = logging.get_logger(__name__)
    
class InsEmbBloomForCausalLM(BloomForCausalLM):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        print('[Child Check] entered Child Class.')
        self.transformer = InsEmbBloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.cos_similarity_record = []
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        instruction: Optional[torch.LongTensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        alpha: Optional[torch.FloatTensor] = 0,
        ins_pool_method: Optional[str] = "mean",
        alpha_rate: Optional[float] = 1.0,
        ins_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import ipdb;ipdb.set_trace()
        assert instruction is not None and ins_attention_mask is not None 
        instruct_outputs = self.transformer(
            input_ids=instruction,
            past_key_values=None,
            attention_mask=ins_attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        instruct_layer_hs = instruct_outputs[-1][:-1]
        instruct_layer_hs =  [item for item in instruct_layer_hs]
        # ins_states = torch.log(torch.clamp(torch.mean(instruct_outputs[0].detach(), dim=1), min=1e-3))
        del instruct_outputs
        if "ins_pool_method" in self.config.__dir__():
            ins_pool_method = self.config.ins_pool_method
        if segment_ids is None:
            segment_ids = token_ids2seg_ids(input_ids, self.config.seg_id)
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            instruct_layer_hs=instruct_layer_hs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            ins_pool_method=ins_pool_method,
            segment_ids=segment_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        del instruct_layer_hs
        
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        output =  CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        segment_ids = None
        if past_key_values:
            segment_ids = token_ids2seg_ids(input_ids, self.config.seg_id)
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # if 'instruction' in kwargs:
            #     kwargs['instruction'] = kwargs['instruction'][:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "instruction": kwargs.get('instruction'),
                    "ins_attention_mask": kwargs.get('ins_attention_mask'),
                    "alpha": kwargs.get("alpha"),
                    "segment_ids": segment_ids
                }
            )
        return model_inputs

class InsEmbBloomModel(BloomModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        selected_layer = config.selected_layer
        ins_fuse_list = []
        ins_transfrom_list = []
        for i in range(config.n_layer):
            if i in selected_layer:
                ins_transfrom_list.append(torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.ins_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.ins_hidden_size, config.hidden_size)
            ))
                ins_fuse_list.append(torch.nn.Linear(2,1, bias=False))

            else:
                ins_transfrom_list.append(None)
                ins_fuse_list.append(None)
        self.ins_fuses = nn.ModuleList(ins_fuse_list)
        # if self.config.add_transform
        self.ins_transforms = nn.ModuleList(ins_transfrom_list)
        self.post_init()


    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape
        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        instruct_layer_hs: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ins_pool_method: Optional[str] = 'mean',
        alpha: Optional[float] = 10.0,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        if "custom_alpha" in self.config.__dir__() and self.config.custom_alpha:
            alpha = self.config.custom_alpha
        input_seg_id = 2
        if "weighted_seg_id" in self.config.__dir__():
            input_seg_id = self.config.weighted_seg_id
        def convert_seg2factor(segment_ids, type='seg_tri'):
            factor_lists= []
            if segment_ids.shape[0]>=1:
                for i in range(segment_ids.shape[0]):
                    segment_id = segment_ids[i:i+1, :]
                    
                    max_seg_id = segment_id.max().item()
                    input_seg_len = (segment_id==input_seg_id).sum().item()
                    factor_list = []
                    if max_seg_id==input_seg_id:
                        input_seg_len = 64

                    for seg_id in range(input_seg_id, max_seg_id+1):
                        if seg_id==max_seg_id and False:
                            factor_list += [0]*((segment_id==seg_id).sum().item())
                        else:
                            if type=='seg_tri':
                                factor_list += [(i)/input_seg_len for i in range((segment_id==seg_id).sum().item())]
                            elif type=='seg_sin':
                                factor_list += [np.sin(np.pi*(i/input_seg_len)) for i in range((segment_id==seg_id).sum().item())]
                            elif type=='sin':
                                input_seg_len = min(input_seg_len, 64)
                                factor_list += [np.sin(np.pi*(i/input_seg_len)) for i in range((segment_id==seg_id).sum().item())]
                    
                    factor_lists.append(factor_list)
            max_len = max([len(ids) for ids in factor_lists])
            for i,ids in enumerate(factor_lists):
                if len(ids)<max_len:
                    factor_lists[i] = [0]*(max_len-len(ids))+ids
            return factor_lists
            


        if instruct_layer_hs is None:
            instruct_layer_hs = [None for _ in range(self.config.n_layer)]
        for i, (block, layer_past, ins_transform, ins_fuse_fn, ins_hs) in enumerate(zip(self.h, past_key_values, self.ins_transforms, self.ins_fuses, instruct_layer_hs)):

            trans_ins_hs = None
            if ins_hs is not None and ins_transform is not None:
                if ins_pool_method=='max':
                    trans_ins_hs = torch.max(ins_transform(ins_hs), dim=-2)[0].unsqueeze(-2)
                elif ins_pool_method=='last_token':
                    trans_ins_hs =  ins_transform(ins_hs[:, -1:, :])
                else:
                    trans_ins_hs = torch.mean(ins_transform(ins_hs), dim=-2).unsqueeze(-2)
                if 'ins_fuse_method' not in self.config.__dir__() or self.config.ins_fuse_method!='fuse':
                    if 'cat_post_ins' not in self.config.__dir__() or str(self.config.cat_post_ins)=='False':
                        hidden_states += alpha * trans_ins_hs
                    else:
                        non_ins_len = hidden_states.shape[1] - ins_hs.shape[1]
                        if self.config.cat_post_ins=='linear':
                            linear_factor = torch.tensor([(i+1)/128 for i in range(non_ins_len)]).reshape(1, -1, 1).to(self.device)
                            hidden_states[:, -(linear_factor.shape[1]):, :] += alpha * trans_ins_hs * linear_factor
                        elif self.config.cat_post_ins=='seg_tri' or self.config.cat_post_ins=='seg_sin':
                            # import ipdb;ipdb.set_trace()
                            linear_factor = torch.tensor(convert_seg2factor(segment_ids, type=self.config.cat_post_ins)).reshape(hidden_states.shape[0], -1, 1).to(self.device)
                            if input_ids.shape[1]==1:
                                linear_factor = linear_factor[:, -1:]
                            hidden_states[:, -(linear_factor.shape[1]):, :] += alpha * trans_ins_hs * linear_factor
                        else:
                            hidden_states[:, ins_hs.shape[1]:, :] += alpha * trans_ins_hs
                else:
                    sent_ins_fuse = torch.stack([hidden_states, trans_ins_hs.repeat(1, hidden_states.shape[1],1)*0.1], dim=-1)
                    hidden_states = ins_fuse_fn(sent_ins_fuse).squeeze(-1)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    

