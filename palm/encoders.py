import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _expand_mask
from transformers import AutoConfig, AutoModel


from pengi import get_audio_encoder
from pengi import Projection


######################################################################################
#  Audio Encoder
######################################################################################
class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int, 
            specaug: bool, mixup: bool, use_pretrained_audioencoder: bool, freeze_audio_encoder_weights: bool,
            use_precomputed_melspec: bool, pretrained_audioencoder_path: str) -> None:
        super().__init__()

        audio_encoder, pretrained_emb_size = get_audio_encoder(audioenc_name)

        if use_pretrained_audioencoder:
            classes_num = 527
            d_in = pretrained_emb_size

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in,
            specaug, mixup, use_precomputed_melspec)

        self.projection = Projection(pretrained_emb_size if use_pretrained_audioencoder else d_in, d_out)

        if freeze_audio_encoder_weights:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output



######################################################################################
#  Text Encoder
######################################################################################
class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int, freeze_text_encoder_weights: bool) -> None:
        super().__init__()

        self.text_model = text_model
        self.base = AutoModel.from_pretrained(text_model)

        if 'clip' in text_model:
            self.clip_text_projection = self.base.text_projection
            self.base = self.base.text_model
            if 'base' in text_model:
                transformer_embed_dim = 512
        
        self.projection = Projection(transformer_embed_dim, d_out)

        if freeze_text_encoder_weights:
            for p in self.base.parameters():
                p.requires_grad = False


    def base_forward_overriden(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        
        output_attentions = output_attentions if output_attentions is not None else self.base.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.base.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.base.config.use_return_dict

        if inputs_embeds is None:
            raise ValueError("You have to specify inputs_embeds")

        bsz, seq_len = inputs_embeds.shape[:2]

        if position_ids is None:
            position_ids = self.base.embeddings.position_ids[:, :seq_len]

        hidden_states = inputs_embeds + self.base.embeddings.position_embedding(position_ids)
        
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.base._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.base.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.base.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(self, x):
        if 'clip' in self.text_model:
            pooled_output = self.base_forward_overriden(**x)[1] # get pooled output
            out = self.clip_text_projection(pooled_output)  # get CLS token output
        else:
            out = self.base(**x)[0]
            out = out[:, 0, :]  # get CLS token output
        
        projected_vec = self.projection(out)
        return projected_vec
