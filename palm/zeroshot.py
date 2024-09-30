import torch
from .encoders import AudioEncoder, TextEncoder



class ZeroShotPENGI(torch.nn.Module):
    def __init__(self, args, pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args
        self.process_text = pengi.preprocess_text

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
        print("ZERO SHOT: loading the weights of the pengi pre-trained audio and text encoders ...")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())


        self.audio_encoder.eval()
        self.text_encoder.eval()
   
        self.device = args.device

        prompt_prefix = args.prompt_prefix
        self.prompts = [f"{prompt_prefix} {class_name}." for class_name in args.classnames]

        print("\n\n################## Zero-Shot PENGI Information ##################")
        print("Prompt Prefix: ", prompt_prefix)
        print("Prompts: ", self.prompts)
        print("###################################################################\n\n")
        
    def forward(self, audio):

        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
 

        tokenized_prompts = self.process_text(self.prompts, enc_tok=True, add_text=False)
        
        prompts_tokens = tokenized_prompts['input_ids'].to(self.device)
        # breakpoint()
        prompts_token_embeddings = self.text_encoder.base.embeddings.token_embedding(prompts_tokens).to(self.device)   # [batch_size, seq_length, embed_dim]
        prompts_attention_mask = tokenized_prompts['attention_mask'].to(self.device)
        
        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = 100.0
        logits = logit_scale * audio_features @ text_features.t()  # logits shape [n_audio_files, n_text_prompts]
        # breakpoint()

        return logits
