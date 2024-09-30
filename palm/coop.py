import torch
import torch.nn as nn


from .encoders import AudioEncoder
from .encoders import TextEncoder



class PromptLearner(nn.Module):
    def __init__(self, args, text_encoder, pengi):
        super().__init__()

        self.args = args
        classnames = args.classnames
        n_cls = len(classnames)

        n_ctx = args.n_ctx # 16
        ctx_dim = args.ctx_dim # 512

        #tokenizer = pengi.enc_tokenizer
        #classnames_token_lens = [len(tokenizer.encode(class_name)) - 2  for class_name in classnames]


        print("Initializing a Generic Context for COOP ...")
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        torch.nn.init.normal_(ctx_vectors, std=0.02)
        ctx = torch.nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {class_name}." for class_name in classnames]

        tokenized_prompts = pengi.preprocess_text(prompts, enc_tok=True, add_text=False)
        
        
        with torch.no_grad():
            tokenized_prompts_embeddings = text_encoder.base.embeddings.token_embedding(tokenized_prompts['input_ids'])   # [batch_size, seq_length, embed_dim]
        

        sos_embedding = tokenized_prompts_embeddings[:,:1,:]
        classnames_and_pad_embedding = tokenized_prompts_embeddings[:,1+n_ctx:,:]

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts
        self.ctx = ctx
        self.register_buffer("sos", sos_embedding)  # SOS
        self.register_buffer("classes_pad", classnames_and_pad_embedding)  # CLASSNAMES, PADDING Token Embeddings


    def forward(self):

        if self.ctx.dim()==2: 
            ctx = self.ctx.unsqueeze(0).expand(self.n_cls,-1,-1)
        else:
            ctx = self.ctx+0.0

        prefix = self.sos
        suffix = self.classes_pad

        prompts_token_embeddings = torch.cat( [prefix, ctx, suffix ], dim=1 )
        
        return self.tokenized_prompts['input_ids'].to(self.args.device), prompts_token_embeddings, self.tokenized_prompts['attention_mask'].to(self.args.device)

class CustomPENGI(nn.Module):
    def __init__(self, args, pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args
        
        pengi_args.specaug = args.spec_aug

        self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path)

        self.text_encoder = TextEncoder(
                    pengi_args.d_proj, 
                    pengi_args.text_model, pengi_args.transformer_embed_dim,
                    pengi_args.freeze_text_encoder_weights)

        # load the weights of the pengi pre-trained audio and text encoders
        print("\n\nCOOP: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.eval()
        self.text_encoder.eval()


        self.prompt_learner = PromptLearner(args, self.text_encoder, pengi)

        self.device = args.device
 

    def forward(self, audio):

        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        prompts_tokens, prompts_token_embeddings, prompts_attention_mask = self.prompt_learner()
        
        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = 100.0
        logits = logit_scale * audio_features @ text_features.t()  # logits shape [n_audio_files, n_text_prompts]
        # breakpoint()

        return logits