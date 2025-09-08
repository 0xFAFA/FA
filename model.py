import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
from torch import autograd
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets


import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, N_CTX = 16, CTX_INIT = "a photo of a", CSC = False, CLASS_TOKEN_POSITION = "end",cfg = None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = N_CTX
        ctx_init = CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = cfg['device']
        

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = ctx_vectors.repeat(n_cls, 1, 1)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype).to(device)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # ----- normal optimized
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # -----


        print(self.ctx.shape)  # 当为"a photo of a"时为[4, 512]，csc为True时为[1000, 4, 512]
        print("------------------")        

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.positional_embedding.device)
        # print('tokenized:',tokenized_prompts.shape) #[len(classnames), 77]

        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            # print('embedding:',embedding.shape)  # [len(classnames), 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        # print('prompts.shape',prompts.shape)  # [len(classnames), ctx_sum, transformer.width] 如[1000, 77, 512]
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.to(dtype = self.dtype, device = prompts.device)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def get_original_text_features(prompt_learner, text_encoder, clip_model):
    device = next(text_encoder.parameters()).device

    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

    with torch.no_grad():
        prompts = prompt_learner()
        tokenized_prompts = prompt_learner.tokenized_prompts

        original_text_features = text_encoder(prompts.cuda(), tokenized_prompts.cuda())

    text_encoder = text_encoder.to(device)
    return original_text_features.to(device)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, 
                 clip_model,
                 CTX_INIT = "a photo of a", 
                 single = False, 
                 n_ctx = 16,
                 csc = False):
        super().__init__()
        
        self.prompt_learner = PromptLearner(classnames, clip_model, CTX_INIT=CTX_INIT, N_CTX=n_ctx, CSC = csc, cfg = cfg)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts 
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.single = single
        self.cfg = cfg
        self.classnum = len(classnames)

        self.text_features_classify_original = get_original_text_features(self.prompt_learner, self.text_encoder, clip_model)

        repeat_sum_num = int(cfg['K']) 

        if repeat_sum_num > 0:
            text_features_classify_original = self.text_features_classify_original.repeat(1, 1)
        else:
            text_features_classify_original = self.text_features_classify_original.repeat(0, 1)
        
        self.text_features_classify_original = text_features_classify_original
        self.repeat_sum_num = repeat_sum_num
        # print(f"self.text_features_classify_original.shape: {self.text_features_classify_original.shape}")

        
    def forward(self, image):        
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))


        prompts = self.prompt_learner().to(image.device)   #[len(classnames), ctx_sum, transformer.width] 如[1000, 77, 512]
        tokenized_prompts = self.tokenized_prompts.to(image.device)  #[len(classnames), 77]
        tokenized_prompts = tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts).type(self.dtype)

        text_features_classify_final = torch.cat((text_features, self.text_features_classify_original), dim=0)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features_classify_final = text_features_classify_final / text_features_classify_final.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features_classify_final.transpose(-1, -2)
        logits_local = logit_scale * local_image_features @ text_features_classify_final.T

        # print(f"logits.shape: {logits.shape}")
        # print(f"logits_local.shape: {logits_local.shape}")
        # print('-------')

        if self.repeat_sum_num - 1 > 0:
            
            batch_size, token_len, sum_template_len = logits_local.shape
            logits_repeat_template = logits.view(batch_size, 2, self.classnum)[:,1,:]
            # print(f"logits_repeat_template.shape: {logits_repeat_template.shape}")
            logits_repeat_context = logits_repeat_template.repeat(1, self.repeat_sum_num - 1)
            # print(f"logits_repeat_context.shape: {logits_repeat_context.shape}")

            logits = torch.cat((logits, logits_repeat_context), dim=1)
            # print(f"logits.shape: {logits.shape}")



            logits_local_repeat_template = logits_local.view(batch_size, token_len, 2, self.classnum)[:,:,1,:]
            # print(f"logits_local_repeat_template.shape: {logits_local_repeat_template.shape}")
            logits_local_repeat_context = logits_local_repeat_template.repeat(1, 1, self.repeat_sum_num - 1)
            # print(f"logits_local_repeat_context.shape: {logits_local_repeat_context.shape}")

            logits_local = torch.cat((logits_local, logits_local_repeat_context), dim=2)
            # print(f"logits_local.shape: {logits_local.shape}")

        
        return logits, logits_local