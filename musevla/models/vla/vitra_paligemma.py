import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Callable
import copy
import numpy as np
import json

from PIL import Image
from functools import partial

from musevla.utils.tensor_utils import move_masked_to_left, get_mask_of_last_masked_index, move_masked_to_left_ids
from musevla.models.vlm_builder import build_vlm

class VITRA_Paligemma(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs=None,
        act_model_configs=None,
        fwd_pred_next_n=1,
        repeated_diffusion_steps:int = 8,
        use_state='DiT',
        use_fov=True,
        use_bf16=False,
        checkpoint_path=None,
        **kwargs,
    ):
        super().__init__()

        self.configs = configs
        self.train_setup_configs = train_setup_configs
        self.act_model_configs = act_model_configs
        self.use_state = use_state
        self.use_fov = use_fov
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.past_action_window_size = 0
        # chunk_size for action prediction
        self.chunk_size = self.configs.get("fwd_pred_next_n", 16)
        self.future_action_window_size = self.chunk_size-1
        self.state_mask_prob = self.configs.get("state_mask_prob", 0.1)
        self.action_type = self.configs['train_dataset'].get("action_type", "angle")
        self.use_state = use_state
        self.use_fov = use_fov
        self.use_bf16 = use_bf16
        if self.action_type == 'angle':
            self.hand_dim = 51
        elif self.action_type == 'keypoints':
            self.hand_dim = 69

        # Initialize the tokenizer and VLM backbone
        self.tokenizer, self.backbone = self._init_backbone(checkpoint_path)
        if self.train_setup_configs is not None and self.train_setup_configs.get("reinit", False):
            initialize_param(self.backbone)

        self.act_model = self._init_act_model()

        if self.use_state == 'VLM':
            self.state_and_mask_dim = 2 * self.configs["state_encoder"]["state_dim"]
            self.vlm_state_encoder = self._init_state_encoder()

        if self.use_fov:
            self.fov_encoder = self._init_fov_encoder()

        # The `cognition_token_id` is set to an unused token ID.
        self.cognition_token_id = self.configs.get("cognition_token_id", 10)
        untied_cognition_token = self.configs.get("untied_cognition_token", True)

        # Use a separately learned `cognition_token_embedding` that does not share parameters with the word embedding matrix
        # if `untied_cognition_token` is True. We initialize this separate `cognition_token_embedding` 
        # with the word embedding parameter corresponding to the specified `cognition_token_id`. 

        if untied_cognition_token:
            print(f"Using separate cognition token")
            print(f"Cognition token id: {self.cognition_token_id}")
            init_id = self.configs.get("cognition_token_init_id", None)
            if init_id is None:
                init_id = self.cognition_token_id
            print(f"Init cognition token with id={init_id}")
            ebd = self.model.get_input_embeddings().weight.data[init_id]
            self.cognition_token = nn.Parameter(ebd.clone())
        else:
            self.cognition_token = None

    def _init_backbone(self, checkpoint_path=None):
        from pathlib import Path
        
        processor, model = build_vlm(self.configs["vlm"])
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        
        # If loading from checkpoint, first resize embeddings to match checkpoint vocab size
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            checkpoint_file_path = Path(checkpoint_path)
            if checkpoint_file_path.is_file():
                checkpoint_dir = checkpoint_file_path.parent
            else:
                checkpoint_dir = checkpoint_file_path
            
            # Try to load tokenizer from checkpoint if available
            tokenizer_path = checkpoint_dir / "tokenizer"
            if tokenizer_path.exists():
                # print(f"Loading tokenizer from {tokenizer_path}")
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.processor.tokenizer = self.tokenizer
                # print(f"Tokenizer loaded successfully with {len(self.tokenizer)} tokens")
                
                # # Resize model embeddings to match loaded tokenizer
                # if len(self.tokenizer) != model.get_input_embeddings().weight.shape[0]:
                #     print(f"Resizing model embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(self.tokenizer)}")
                #     model.resize_token_embeddings(len(self.tokenizer))
            else:
                # No tokenizer saved, but checkpoint might have extra embedding weights
                # Check checkpoint to infer vocab size
                print(f"No tokenizer found at {tokenizer_path}, checking checkpoint vocab size...")
                print(f"Loading checkpoint from {checkpoint_path} to check vocab size...")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Debug: print some checkpoint keys
                checkpoint_keys = list(checkpoint.keys())
                # print(f"Checkpoint has {len(checkpoint_keys)} keys. First few keys: {checkpoint_keys[:5]}")
                
                if "backbone.language_model.model.embed_tokens.weight" in checkpoint:
                    checkpoint_vocab_size = checkpoint["backbone.language_model.model.embed_tokens.weight"].shape[0]
                    current_vocab_size = model.get_input_embeddings().weight.shape[0]
                    
                    # print(f"Checkpoint has {checkpoint_vocab_size} tokens, current model has {current_vocab_size}")
                    
                    if checkpoint_vocab_size > current_vocab_size:
                        print(f"Resizing model embeddings to match checkpoint: {checkpoint_vocab_size}")
                        
                        # Resize to match checkpoint
                        model.resize_token_embeddings(checkpoint_vocab_size)
                        
                        # Add placeholder tokens to tokenizer to match the vocab size
                        num_extra_tokens = checkpoint_vocab_size - current_vocab_size
                        extra_tokens = [f"<extra_id_{i}>" for i in range(num_extra_tokens)]
                        self.tokenizer.add_tokens(extra_tokens)
                        print(f"Added {num_extra_tokens} placeholder tokens to tokenizer")
                    elif checkpoint_vocab_size == current_vocab_size:
                        pass
                        # print(f"Checkpoint and model have same vocab size: {checkpoint_vocab_size}")
                    else:
                        print(f"Warning: Checkpoint has fewer tokens ({checkpoint_vocab_size}) than model ({current_vocab_size})")
                else:
                    print(f"Warning: Could not find 'backbone.language_model.model.embed_tokens.weight' in checkpoint")
                    # Try to find similar keys
                    embed_keys = [k for k in checkpoint_keys if 'embed' in k.lower()]
                    print(f"Keys containing 'embed': {embed_keys[:10]}")
                
                # Free memory
                del checkpoint
        
        # Add special sensor tokens for VLM finetuning if not already present
        special_tokens = ["<None>", "<thermal>", "<radar>", "<audio>"]
        existing_special_tokens = [token for token in special_tokens if token in self.tokenizer.get_vocab()]
        
        if len(existing_special_tokens) < len(special_tokens):
            new_special_tokens = [token for token in special_tokens if token not in existing_special_tokens]
            num_added_tokens = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_special_tokens}
            )
            
            if num_added_tokens > 0:
                print(f"Added {num_added_tokens} special tokens to tokenizer: {new_special_tokens}")
                
                # Only resize if current model embeddings are smaller than tokenizer
                current_embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(self.tokenizer) > current_embedding_size:
                    print(f"Resizing model embeddings from {current_embedding_size} to {len(self.tokenizer)}")
                    model.resize_token_embeddings(len(self.tokenizer))
                    
                    # Initialize new token embeddings with small random values
                    with torch.no_grad():
                        new_embeddings = model.get_input_embeddings().weight[-num_added_tokens:]
                        nn.init.normal_(new_embeddings, mean=0.0, std=0.02)
                else:
                    print(f"Model already has sufficient embedding size ({current_embedding_size}), no resize needed")
                    # Initialize the embeddings for the new tokens from existing embeddings
                    with torch.no_grad():
                        for i, token in enumerate(new_special_tokens):
                            token_id = self.tokenizer.convert_tokens_to_ids(token)
                            # Initialize from a similar token (just use the mean of existing embeddings)
                            model.get_input_embeddings().weight[token_id] = model.get_input_embeddings().weight[:-num_added_tokens].mean(dim=0)
                            nn.init.normal_(model.get_input_embeddings().weight[token_id].unsqueeze(0), mean=0.0, std=0.02)
        
        # Store sensor token IDs for later use
        self.sensor_token_ids = {
            "none": self.tokenizer.convert_tokens_to_ids("<None>"),
            "thermal": self.tokenizer.convert_tokens_to_ids("<thermal>"),
            "radar": self.tokenizer.convert_tokens_to_ids("<radar>"),
            "audio": self.tokenizer.convert_tokens_to_ids("<audio>"),
        }
        
        # DON'T set add_eos_token globally - let dataset/evaluation control it per-use
        # This prevents unwanted EOS tokens in prompts during training
        # if hasattr(self.tokenizer, 'add_eos_token'):
        #     print(f"Tokenizer add_eos_token: {self.tokenizer.add_eos_token} (will be managed per-use)")
        
        # print(f"Final tokenizer size: {len(self.tokenizer)}, model embedding size: {model.get_input_embeddings().weight.shape[0]}")
        
        return self.tokenizer, model

    def _init_fov_encoder(self):
        from musevla.utils.nn_utils import MLPProjector
        fov_dim = 2 # fov_x, fov_y
        mlp = MLPProjector(fov_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_state_encoder(self):
        from musevla.utils.nn_utils import MLPProjector
        mlp = MLPProjector(self.state_and_mask_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_act_model(self):
        from musevla.models.action_model.diffusion_policy import DiffusionPolicy
        action_head = DiffusionPolicy(
            model_type = self.act_model_configs.get("model_type", 'DiT-B'),
            token_size = self.act_model_configs.get("token_size", -1),
            in_channels = self.act_model_configs.get("action_dim", 192),
            future_action_window_size = self.future_action_window_size,
            past_action_window_size = self.past_action_window_size,
            use_state = self.use_state,
            action_type = self.configs['train_dataset'].get("action_type", "angle"),
            state_dim = self.configs["state_encoder"]["state_dim"] if self.use_state=='DiT' else None,
            loss_type = self.configs.get("loss_type", "human"),
        )

        for param in action_head.parameters():
            assert param.dtype == torch.float32, f"Loaded diffusion action model parameter not in full precision: {param}"

        return action_head

    def trainable_params_setup(self):
        model = self.model
        model.config.use_cache = False

        if self.train_setup_configs.get("freeze_option", "full_finetune") == "full_finetune":
            model.requires_grad_(True)
            self.vision_tower.requires_grad_(True)
            self.word_embedding.requires_grad_(True)

        if self.train_setup_configs.get("freeze_option", "only_head_and_token") == "only_head_and_token":
            model.requires_grad_(False)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(False)

        if self.train_setup_configs.get("freeze_option", "freeze_vision_encoder") == "freeze_vision_encoder":
            model.requires_grad_(True)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(True)

        if self.act_model is not None:
            self.act_model.requires_grad_(True)

        if self.use_state == 'VLM':
            self.vlm_state_encoder.requires_grad_(True)
        
        if self.use_fov:
            self.fov_encoder.requires_grad_(True)

        if self.cognition_token is not None:
            self.cognition_token.requires_grad_(True)

    @property
    def image_processor(self):
        return self.model.processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def model(self):
        return self.backbone

    def _forward_act_model(
        self,
        vlm_features: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        mode: str = "train",
        repeated_diffusion_steps: int = 1,
        cfg_scale: float = 5.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        **kwargs,
    ):
        
        actions = None
        action_loss = None

        B = vlm_features.shape[0]
        action_features = self.extract_cognition_token(vlm_features, attention_mask) #[B, D]
        model_dtype = next(self.act_model.net.parameters()).dtype
        action_features = action_features.to(model_dtype)
        
        action_features_repeated = action_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_masks_repeated = action_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)

        action_features_repeated = action_features_repeated.view(B*repeated_diffusion_steps, 1, action_features.shape[-1])
        action_masks_repeated = action_masks_repeated.view(B*repeated_diffusion_steps, action_masks.shape[1], action_masks.shape[2])

        if self.use_state == 'DiT':
            current_state_repeated = current_state.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_repeated = current_state_repeated.view(B*repeated_diffusion_steps, 1, current_state.shape[1])
            current_state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_mask_repeated = current_state_mask_repeated.view(B*repeated_diffusion_steps, 1, current_state_mask.shape[1])
        else:
            current_state_repeated = None
            current_state_mask_repeated = None

        if mode == "train":
            actions_repeated = action_labels.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            actions_repeated = actions_repeated.view(B*repeated_diffusion_steps, action_labels.shape[1], action_labels.shape[2])
            if self.use_state == 'DiT':
                action_loss = self.act_model.loss(actions_repeated, action_features_repeated, action_masks_repeated, current_state_repeated, current_state_mask_repeated)
            else:
                action_loss = self.act_model.loss(actions_repeated, action_features_repeated, action_masks_repeated)
            
            # If debug requested, also sample predicted actions for comparison
            debug_training = kwargs.get('debug_training', False)
            if debug_training:
                with torch.no_grad():
                    actions = self.act_model.sample(
                        action_features_repeated[:B],  # Only need one set (no repeated)
                        5.0,  # cfg_scale
                        current_state_repeated[:B] if current_state_repeated is not None else None,
                        current_state_mask_repeated[:B] if current_state_mask_repeated is not None else None,
                        True,  # use_ddim
                        10,    # num_ddim_steps
                        action_masks_repeated[:B],
                    )
            
            return actions, action_loss
        else:
            # evaluate mode
            # sample multiple action chunks
            actions = self.act_model.sample(
                action_features_repeated,
                cfg_scale,
                current_state_repeated,      #[B*sample_times, 1, D]
                current_state_mask_repeated, #[B*sample_times, 1, D]
                use_ddim,
                num_ddim_steps,
                action_masks_repeated, # ori x_mask
            )

            return actions, action_loss

    def extract_cognition_token(self, output_hs, attention_mask):
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, output_hs.size(-1))
        action_features = output_hs.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        return action_features

    def prepare_vlm_input_embeddings(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        skip_cognition_token: bool = False,
        skip_move_masked: bool = False,
        debug_training: bool = False,
        **kwargs
    ):
        B = input_ids.shape[0]
        
        # Debug: Check for invalid input_ids
        if debug_training:
            max_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            invalid_ids = (input_ids >= max_vocab_size) | (input_ids < 0)
            if invalid_ids.any():
                print(f"[EMBED DEBUG ERROR] Found {invalid_ids.sum().item()} invalid input IDs!")
                print(f"[EMBED DEBUG ERROR] Vocab size: {max_vocab_size}, invalid IDs: {input_ids[invalid_ids].unique().tolist()}")
        
        word_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Debug: Check word embeddings
        if debug_training:
            has_nan = torch.isnan(word_embeds).any().item()
            has_inf = torch.isinf(word_embeds).any().item()
            if has_nan or has_inf:
                print(f"[EMBED DEBUG ERROR] word_embeds has NaN: {has_nan}, Inf: {has_inf}")
        
        input_ids_mask = attention_mask 

        cog_ids = torch.ones_like(input_ids[:, 0:1]) * self.cognition_token_id # [B, 1]
        cog_embeds = self.model.get_input_embeddings()(cog_ids)
        cog_ids_mask = torch.ones_like(cog_ids, dtype=torch.bool)

        if hasattr(self, 'cognition_token') and self.cognition_token is not None:
            assert self.cognition_token.shape[0] == cog_embeds.shape[-1], f"cognition token shape {self.cognition_token.shape} does not match cog embeds shape {cog_embeds.shape}"
            cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(B, -1, -1) 

        # Build the list of embeddings and masks to concatenate
        embeds_list = [word_embeds]
        masks_list = [input_ids_mask]
        num_additional_tokens = 0

        if self.use_state == 'VLM':
            current_state = current_state * current_state_mask.to(current_state.dtype)
            state_embeds = self.state_encoder(torch.cat([current_state, current_state_mask.to(current_state.dtype)], dim=1))
            state_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)
        
            embeds_list.append(state_embeds.unsqueeze(1))
            masks_list.append(state_ids_mask)
            num_additional_tokens += 1

        if self.use_fov:
            fov_embeds = self.fov_encoder(fov)
            fov_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)

            embeds_list.append(fov_embeds.unsqueeze(1))
            masks_list.append(fov_ids_mask)
            num_additional_tokens += 1

        # Append cognition token at the end (skip for VLM-only generation)
        if not skip_cognition_token:
            embeds_list.append(cog_embeds)
            masks_list.append(cog_ids_mask)
            num_additional_tokens += 1

        # Concatenate all
        inputs_embeds = torch.cat(embeds_list, dim=1)
        inputs_masks = torch.cat(masks_list, dim=1)

        # Note: Here we only use `self.cognition_token_id` as a placeholder for the token corresponding to the FOV or states (if any) input. 
        # In practice, the embedding passed to the LLM will be replaced with the actual FOV or states (if any) embedding.
        additional_tokens = torch.full((B, num_additional_tokens), self.cognition_token_id, dtype=input_ids.dtype, device=input_ids.device)

        # Skip move_masked_to_left for VLM finetuning to preserve sequence length for label alignment
        if skip_move_masked:
            # Keep original length - don't remove padding tokens
            # NOTE: Don't add additional mask tokens - embeddings are already concatenated above!
            # The inputs_embeds already includes cognition token from embeds_list
            # Just update input_ids for reference (not used in forward pass with inputs_embeds)
            input_ids = torch.cat([input_ids, additional_tokens], dim=1)
            # inputs_masks already has correct length from concatenating masks_list above
        else:
            # VLA mode - remove padding tokens to reduce computation
            inputs_embeds, attention_mask = move_masked_to_left(inputs_embeds, inputs_masks)
            input_ids = torch.cat([input_ids, additional_tokens], dim=1)
            input_ids, inputs_masks = move_masked_to_left_ids(input_ids, inputs_masks)

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        # Merge text and images
        if pixel_values is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                image_features = self.model.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self.model._update_causal_mask(
            attention_mask, None, None, cache_position, input_ids, inputs_embeds, False
        )
        return {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
        }, inputs_masks

    def prepare_vlm_input_embeddings_two_images(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        Prepare embeddings for Stage 2 (VLA) with dual images (RGB + sensor).
        
        Args:
            pixel_values: Dual images [B, 2, C, H, W] where [:, 0] is RGB and [:, 1] is sensor
            input_ids: Input token IDs (prompt only)
            labels: Label token IDs (answer)
            attention_mask: Attention mask for input_ids
            current_state_mask: Optional state mask
            current_state: Optional current state
            fov: Optional field of view
            
        Returns:
            stage2_inputs: Dict with attention_mask, position_ids, inputs_embeds, cache_position
            stage2_masks: Attention mask after padding removal
        """
        debug_training = kwargs.get('debug_training', False)
        
        # Merge input_ids and labels to get full sequence [prompt | answer | pad]
        # input_ids contains: [prompt | pad]
        # labels contains: [-100 for prompt | answer | -100 for pad]
        # We need: [prompt | answer | pad]
        B = input_ids.shape[0]
        
        # Build full sequence by replacing label positions with actual answer tokens
        full_input_ids = input_ids.clone()
        answer_mask = labels != -100
        full_input_ids[answer_mask] = labels[answer_mask]
        
        if debug_training:
            # Analyze the merged sequence
            prompt_mask = (labels == -100) & (attention_mask == 1)
            answer_mask_count = answer_mask[0].sum().item()
            prompt_length = prompt_mask[0].sum().item()
            padding_mask = full_input_ids == self.tokenizer.pad_token_id
            padding_count = padding_mask[0].sum().item()
            
            print(f"\n[STAGE2 DEBUG] === Sequence Composition (sample 0) ===")
            print(f"[STAGE2 DEBUG] Prompt tokens: {prompt_length}")
            print(f"[STAGE2 DEBUG] Answer tokens: {answer_mask_count}")
            print(f"[STAGE2 DEBUG] Padding tokens: {padding_count}")
            print(f"[STAGE2 DEBUG] Total: {full_input_ids.shape[1]}")
            
            # Decode sections
            if prompt_length > 0:
                prompt_ids = full_input_ids[0][prompt_mask[0]]
                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                print(f"[STAGE2 DEBUG] Prompt (first 100 chars): '{prompt_text[:100]}'")
            
            if answer_mask_count > 0:
                answer_ids = labels[0][answer_mask[0]]
                answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=False)
                print(f"[STAGE2 DEBUG] Answer: '{answer_text}'")
        
        # Find where valid content ends (last non-padding position) for each sample
        # We'll insert cognition token right after the answer, before padding
        is_not_pad = full_input_ids != self.tokenizer.pad_token_id
        last_valid_pos = is_not_pad.long().cumsum(dim=1).argmax(dim=1)  # [B]
        
        # Create new sequence with cognition token inserted after valid content
        # Format: [256 extra_image_tokens | prompt | answer | cognition_token | padding]
        # Note: PaliGemma tokenizes one image as 256 tokens, so we need 256 extra tokens for the second image
        image_token_id = self.model.config.image_token_index
        cog_token_id = self.cognition_token_id
        seq_len = full_input_ids.shape[1]
        num_extra_image_tokens = 256  # PaliGemma image patch tokens
        
        # Build new sequence with 256 extra image tokens at start and cognition token after answer
        new_full_input_ids = torch.full((B, seq_len + num_extra_image_tokens + 1), self.tokenizer.pad_token_id, 
                                        dtype=full_input_ids.dtype, device=full_input_ids.device)
        
        for i in range(B):
            valid_len = (last_valid_pos[i] + 1).item()
            # Layout: [256 extra_image_tokens | valid_content | cognition | padding]
            new_full_input_ids[i, :num_extra_image_tokens] = image_token_id
            new_full_input_ids[i, num_extra_image_tokens:num_extra_image_tokens+valid_len] = full_input_ids[i, :valid_len]
            new_full_input_ids[i, num_extra_image_tokens+valid_len] = cog_token_id
        
        full_input_ids = new_full_input_ids
        
        # Create attention mask for full sequence (1 for valid tokens including cognition, 0 for padding)
        full_attention_mask = torch.ones_like(full_input_ids, dtype=torch.bool)
        full_attention_mask[full_input_ids == self.tokenizer.pad_token_id] = False
        
        if debug_training:
        #     # Verify the final sequence structure
            sample_0_ids = full_input_ids[0]
            
            # Decode the last part of the sequence to see prompt->answer->cognition->padding
            last_point = -70
            
            print(f"\n[STAGE2 DEBUG] === Decoded Sequence Structure ===")
            # Decode from start (after first image) to show full structure
            
            # Show last half starting from middle
            print(f"\n[STAGE2 DEBUG] Last from position {last_point}:")
            last_half_ids = sample_0_ids[last_point:]
            decoded_last_half = self.tokenizer.decode(last_half_ids, skip_special_tokens=False)
            print(f"[STAGE2 DEBUG] '{decoded_last_half}'")
        
        # Process dual images separately (PaliGemma expects [B, C, H, W])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            rgb_features = self.model.get_image_features(pixel_values[:, 0, :, :, :])
            sensor_features = self.model.get_image_features(pixel_values[:, 1, :, :, :])
        
        # Manually build Stage 2 embeddings (similar to prepare_vlm_input_embeddings but for dual images)
        # Get word embeddings for full sequence
        word_embeds = self.model.get_input_embeddings()(full_input_ids)
        
        # Combine RGB and sensor features [B, num_image_tokens * 2, hidden_dim]
        combined_image_features = torch.cat([rgb_features, sensor_features], dim=1)
        
        # Replace image token positions with combined features
        special_image_mask = (full_input_ids == self.model.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(word_embeds)
        
        # Inject combined image features into word embeddings
        inputs_embeds = word_embeds.masked_scatter(
            special_image_mask,
            combined_image_features.to(word_embeds.device, word_embeds.dtype)
        )
        
        # Replace cognition token embedding with learned embedding    
        cog_mask = (full_input_ids == self.cognition_token_id).unsqueeze(-1).expand_as(word_embeds)
        if hasattr(self, 'cognition_token') and self.cognition_token is not None:
            cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
            # Find cognition token positions and replace
            for i in range(B):
                cog_pos = (full_input_ids[i] == self.cognition_token_id).nonzero(as_tuple=True)[0]
                if len(cog_pos) > 0:
                    inputs_embeds[i, cog_pos[0]] = cog_embeds[i, 0]
        
        # Keep padding tokens (no need to remove since sequence is complete)
        # Unlike VLA mode with separate prompt/answer, here we have full sequence already
        stage2_masks = full_attention_mask

        # Prepare position_ids and cache_position
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        
        # Create causal mask (input_ids and embeddings now have matching lengths)
        causal_mask = self.model._update_causal_mask(
            stage2_masks, None, None, cache_position, 
            full_input_ids, inputs_embeds, False
        )
        
        return {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
        }, stage2_masks

    def prepare_vlm_input_embeddings_all_sensor_images(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        Prepare embeddings for multi-sensor VLA with all sensor images (RGB + thermal + radar + acoustic).
        
        Args:
            pixel_values: All sensor images [B, 4, C, H, W] where:
                          [:, 0] is RGB
                          [:, 1] is thermal
                          [:, 2] is radar  
                          [:, 3] is acoustic
            input_ids: Input token IDs (prompt only)
            labels: Label token IDs (answer)
            attention_mask: Attention mask for input_ids
            current_state_mask: Optional state mask
            current_state: Optional current state
            fov: Optional field of view
            
        Returns:
            stage2_inputs: Dict with attention_mask, position_ids, inputs_embeds, cache_position
            stage2_masks: Attention mask after padding removal
        """
        debug_training = kwargs.get('debug_training', False)
        
        # Merge input_ids and labels to get full sequence [prompt | answer | pad]
        B = input_ids.shape[0]
        
        # Build full sequence by replacing label positions with actual answer tokens
        full_input_ids = input_ids.clone()
        answer_mask = labels != -100
        full_input_ids[answer_mask] = labels[answer_mask]
        
        if debug_training:
            # Analyze the merged sequence
            prompt_mask = (labels == -100) & (attention_mask == 1)
            answer_mask_count = answer_mask[0].sum().item()
            prompt_length = prompt_mask[0].sum().item()
            padding_mask = full_input_ids == self.tokenizer.pad_token_id
            padding_count = padding_mask[0].sum().item()
            
            print(f"\n[ALL_SENSOR DEBUG] === Sequence Composition (sample 0) ===")
            print(f"[ALL_SENSOR DEBUG] Prompt tokens: {prompt_length}")
            print(f"[ALL_SENSOR DEBUG] Answer tokens: {answer_mask_count}")
            print(f"[ALL_SENSOR DEBUG] Padding tokens: {padding_count}")
            print(f"[ALL_SENSOR DEBUG] Total: {full_input_ids.shape[1]}")
        
        # Find where valid content ends (last non-padding position) for each sample
        is_not_pad = full_input_ids != self.tokenizer.pad_token_id
        last_valid_pos = is_not_pad.long().cumsum(dim=1).argmax(dim=1)  # [B]
        
        # Create new sequence with cognition token inserted after valid content
        # Format: [256*3 extra_image_tokens | prompt | answer | cognition_token | padding]
        # Note: PaliGemma tokenizes one image as 256 tokens, so we need 256*3 extra tokens for 3 additional sensor images
        image_token_id = self.model.config.image_token_index
        cog_token_id = self.cognition_token_id
        seq_len = full_input_ids.shape[1]
        num_extra_image_tokens = 256 * 3  # 3 additional sensor images (thermal, radar, acoustic)
        
        # Build new sequence with extra image tokens at start and cognition token after answer
        new_full_input_ids = torch.full((B, seq_len + num_extra_image_tokens + 1), self.tokenizer.pad_token_id, 
                                        dtype=full_input_ids.dtype, device=full_input_ids.device)
        
        for i in range(B):
            valid_len = (last_valid_pos[i] + 1).item()
            # Layout: [768 extra_image_tokens | valid_content | cognition | padding]
            new_full_input_ids[i, :num_extra_image_tokens] = image_token_id
            new_full_input_ids[i, num_extra_image_tokens:num_extra_image_tokens+valid_len] = full_input_ids[i, :valid_len]
            new_full_input_ids[i, num_extra_image_tokens+valid_len] = cog_token_id
        
        full_input_ids = new_full_input_ids
        
        # Create attention mask for full sequence
        full_attention_mask = torch.ones_like(full_input_ids, dtype=torch.bool)
        full_attention_mask[full_input_ids == self.tokenizer.pad_token_id] = False
        
        if debug_training:
            sample_0_ids = full_input_ids[0]
            last_point = -70
            
            print(f"\n[ALL_SENSOR DEBUG] === Decoded Sequence Structure ===")
            print(f"\n[ALL_SENSOR DEBUG] Last from position {last_point}:")
            last_half_ids = sample_0_ids[last_point:]
            decoded_last_half = self.tokenizer.decode(last_half_ids, skip_special_tokens=False)
            print(f"[ALL_SENSOR DEBUG] '{decoded_last_half}'")
        
        # Process all 4 images separately (PaliGemma expects [B, C, H, W])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            rgb_features = self.model.get_image_features(pixel_values[:, 0, :, :, :])
            thermal_features = self.model.get_image_features(pixel_values[:, 1, :, :, :])
            radar_features = self.model.get_image_features(pixel_values[:, 2, :, :, :])
            acoustic_features = self.model.get_image_features(pixel_values[:, 3, :, :, :])
        
        # Combine all image features [B, num_image_tokens * 4, hidden_dim]
        combined_image_features = torch.cat([rgb_features, thermal_features, radar_features, acoustic_features], dim=1)
        
        # Get word embeddings for full sequence
        word_embeds = self.model.get_input_embeddings()(full_input_ids)
        
        # Replace image token positions with combined features
        special_image_mask = (full_input_ids == self.model.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(word_embeds)
        
        # Inject combined image features into word embeddings
        inputs_embeds = word_embeds.masked_scatter(
            special_image_mask,
            combined_image_features.to(word_embeds.device, word_embeds.dtype)
        )
        
        # Replace cognition token embedding with learned embedding    
        cog_mask = (full_input_ids == self.cognition_token_id).unsqueeze(-1).expand_as(word_embeds)
        if hasattr(self, 'cognition_token') and self.cognition_token is not None:
            cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
            # Find cognition token positions and replace
            for i in range(B):
                cog_pos = (full_input_ids[i] == self.cognition_token_id).nonzero(as_tuple=True)[0]
                if len(cog_pos) > 0:
                    inputs_embeds[i, cog_pos[0]] = cog_embeds[i, 0]
        
        # Keep padding tokens (no need to remove since sequence is complete)
        stage2_masks = full_attention_mask

        # Prepare position_ids and cache_position
        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        
        # Create causal mask
        causal_mask = self.model._update_causal_mask(
            stage2_masks, None, None, cache_position, 
            full_input_ids, inputs_embeds, False
        )
        
        return {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
        }, stage2_masks

    def prepare_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):

        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
                pixel_values,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                **kwargs,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model.language_model(
                past_key_values=None,
                use_cache=use_cache,
                output_hidden_states=True,
                num_logits_to_keep=0, # can be modified
                **vlm_inputs
            )

        output_hs = outputs.hidden_states[-1]
        return output_hs, inputs_masks

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        mode="train",
        task_mode="vla",
        labels: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):

        assert mode == "train", f"mode {mode} not supported in the forward function."

        loss = {}
        
        # VLM-only finetuning mode
        if task_mode == "vlm_finetune":
            # Prepare inputs for VLM
            debug_training = kwargs.get('debug_training', False)
            vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
                pixel_values,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                skip_move_masked=True,
                skip_cognition_token=True,
                debug_training=debug_training,
            )
            

            # Pass token_type_ids and labels to PaliGemma so is_training=True
            # This enables prefix-LM masking: bidirectional on prompt, causal on answer
            # PaliGemma also computes loss internally (fp32 upcast), so we use it directly
            outputs = self.model(
                attention_mask=inputs_masks,
                position_ids=vlm_inputs["position_ids"],
                inputs_embeds=vlm_inputs["inputs_embeds"],
                labels=labels,
                token_type_ids=token_type_ids,
                use_cache=use_cache,
            )
            
            # Cast logits to fp32 for numerically stable cross-entropy over large vocab
            with torch.cuda.amp.autocast(enabled=False):
                logits = outputs.logits.float()  # [batch_size, seq_len, vocab_size]
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten for cross-entropy
                shift_logits = shift_logits.view(-1, logits.size(-1))  # [batch*(seq-1), vocab]
                shift_labels = shift_labels.view(-1)  # [batch*(seq-1)]
                
                # Compute cross-entropy loss (ignore -100 labels)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                vlm_loss_value = loss_fct(shift_logits, shift_labels)
            
            
            # Extract language modeling loss
            vlm_loss = vlm_loss_value
            loss['vlm_loss'] = vlm_loss
            
            # Debug: Per-token loss and greedy generation test
            if debug_training:
                try:
                    with torch.no_grad():
                        outputs_with_logits = self.model(
                            attention_mask=inputs_masks[:1],
                            position_ids=vlm_inputs["position_ids"][:1],
                            inputs_embeds=vlm_inputs["inputs_embeds"][:1],
                            labels=labels[:1],
                            token_type_ids=token_type_ids[:1] if token_type_ids is not None else None,
                            use_cache=False,
                        )
                    
                    logits_dbg = outputs_with_logits.logits[0]  # [seq_len, vocab_size]
                    
                    # Get positions where we have labels
                    label_mask = labels[0] != -100
                    label_positions = torch.where(label_mask)[0]
                    expected_tokens = labels[0][label_mask]
                    
                    if len(label_positions) > 0:
                        # In causal LM, logits[i] predicts token at position i+1,
                        # so to get the prediction for labels[p], use logits[p-1].
                        predicted_tokens = logits_dbg[label_positions - 1].argmax(dim=-1)
                        
                        first_label_pos = label_positions[0].item()
                        last_label_pos = label_positions[-1].item()
                        seq_len_dbg = logits_dbg.shape[0]
                        
                        # Context before answer
                        context_before_start = max(0, first_label_pos - 40)
                        context_after_end = min(seq_len_dbg, last_label_pos + 11)
                        
                        full_input_ids = input_ids[0]
                        context_before_ids = full_input_ids[context_before_start:first_label_pos]
                        context_after_ids = full_input_ids[last_label_pos + 1:context_after_end]
                        
                        print(f"\n[VLM_FINETUNE DEBUG] ========== Batch Debug ==========")
                        
                        if len(context_before_ids) > 0:
                            context_before_text = self.tokenizer.decode(context_before_ids, skip_special_tokens=False)
                            print(f"\n[VLM_FINETUNE DEBUG] === CONTEXT BEFORE ANSWER ===")
                            print(f"[VLM_FINETUNE DEBUG] Expected: '{context_before_text}'")
                        
                        expected_text = self.tokenizer.decode(expected_tokens, skip_special_tokens=False)
                        predicted_text = self.tokenizer.decode(predicted_tokens, skip_special_tokens=False)
                        
                        print(f"\n[VLM_FINETUNE DEBUG] === ANSWER TOKENS ===")
                        print(f"[VLM_FINETUNE DEBUG] Positions [{first_label_pos}:{last_label_pos+1}]")
                        print(f"[VLM_FINETUNE DEBUG] Expected: '{expected_text}'")
                        print(f"[VLM_FINETUNE DEBUG] Predicted: '{predicted_text}'")
                        print(f"[VLM_FINETUNE DEBUG] VLM Loss: {vlm_loss.item():.6f}")
                        
                        if len(context_after_ids) > 0:
                            context_after_text = self.tokenizer.decode(context_after_ids, skip_special_tokens=False)
                            print(f"\n[VLM_FINETUNE DEBUG] === CONTEXT AFTER ANSWER ===")
                            print(f"[VLM_FINETUNE DEBUG] Expected: '{context_after_text}'")
                        
                        print(f"\n[VLM_FINETUNE DEBUG] === FULL SEQUENCE INFO ===")
                        print(f"[VLM_FINETUNE DEBUG] Total sequence length: {full_input_ids.shape[0]}")
                        print(f"[VLM_FINETUNE DEBUG] First label at: {first_label_pos}, Last label at: {last_label_pos}")
                        
                        print(f"[VLM_FINETUNE DEBUG] ========== End Batch Debug ==========\n")
                    else:
                        print(f"[VLM_FINETUNE DEBUG] No label positions found to check")
                        
                except Exception as e:
                    print(f"[VLM_FINETUNE DEBUG] Logit check failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            return loss

        # VLM-VLA co-training mode
        elif task_mode == "vlm_vla_cofinetune":
            # Extract loss weights from config
            vlm_loss_weight = kwargs.get('vlm_loss_weight', 1.0)
            action_loss_weight = kwargs.get('action_loss_weight', 1.0)
            log_loss_magnitudes = kwargs.get('log_loss_magnitudes', False)
            
            
            # Stage 1: VLM sensor token prediction
            # Process RGB image only (first channel of dual images)
            rgb_pixel_values = pixel_values[:, 0, :, :, :]  # [B, C, H, W]
            
            # Prepare VLM inputs (skip cognition token for VLM-only generation)
            vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
                rgb_pixel_values,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                skip_move_masked=True,
                skip_cognition_token=True,
            )

            # Debug: Check supervised tokens
            debug_training = kwargs.get('debug_training', False)
            if debug_training:
                num_supervised = (labels != -100).sum().item()
                batch_size = labels.shape[0]
                print(f"\n[TRAINING DEBUG] ========== Batch Debug ==========")

            # Forward pass for VLM with causal teacher forcing
            # Pass token_type_ids and labels so PaliGemma uses prefix-LM masking
            vlm_outputs = self.model(
                attention_mask=inputs_masks,
                position_ids=vlm_inputs["position_ids"],
                inputs_embeds=vlm_inputs["inputs_embeds"],
                labels=labels,
                token_type_ids=token_type_ids,
                use_cache=use_cache,
            )
            
            # Cast logits to fp32 for numerically stable cross-entropy over large vocab
            with torch.cuda.amp.autocast(enabled=False):
                logits = vlm_outputs.logits.float()  # [B, seq_len, vocab_size]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                vlm_loss = loss_fct(shift_logits, shift_labels)
                vlm_outputs.loss = vlm_loss

            # Extract language modeling loss
            vlm_loss = vlm_outputs.loss
            loss['vlm_loss'] = vlm_loss
            
            # Debug: Per-token loss and greedy generation test
            if debug_training:
                # Greedy generation test (only on first sample to avoid slowdown)
                # NOTE: Skip generation test when using FSDP as it can corrupt model state
                try:
                    # Get logits from the forward pass
                    with torch.no_grad():
                        # Re-run forward to get logits
                        outputs_with_logits = self.model(
                            attention_mask=inputs_masks[:1],
                            position_ids=vlm_inputs["position_ids"][:1],
                            inputs_embeds=vlm_inputs["inputs_embeds"][:1],
                            labels=labels[:1],
                            token_type_ids=token_type_ids[:1] if token_type_ids is not None else None,
                            use_cache=False,
                        )
                    
                    logits = outputs_with_logits.logits[0]  # [seq_len, vocab_size]
                    
                    # Get positions where we have labels
                    label_mask = labels[0] != -100
                    label_positions = torch.where(label_mask)[0]
                    expected_tokens = labels[0][label_mask]
                    
                    if len(label_positions) > 0:
                        # Check argmax at each label position
                        # NOTE: In causal LM, logits[i] predicts token at position i+1,
                        # so to get the prediction for labels[p], use logits[p-1].
                        predicted_tokens = logits[label_positions - 1].argmax(dim=-1)
                        
                        # Get context: 5 tokens before and after
                        first_label_pos = label_positions[0].item()
                        last_label_pos = label_positions[-1].item()
                        seq_len = logits.shape[0]
                        
                        # Get predicted tokens for 5 positions before and after label region
                        context_before_start = max(0, first_label_pos - 5)
                        context_after_end = min(seq_len, last_label_pos + 6)
                        
                        # Get predictions for context regions
                        if context_before_start < first_label_pos:
                            context_before_pred = logits[context_before_start:first_label_pos].argmax(dim=-1)
                        else:
                            context_before_pred = torch.tensor([], dtype=torch.long)
                        
                        if last_label_pos + 1 < context_after_end:
                            context_after_pred = logits[last_label_pos + 1:context_after_end].argmax(dim=-1)
                        else:
                            context_after_pred = torch.tensor([], dtype=torch.long)
                        
                        # Get input_ids for context (use the input_ids from function parameters)
                        full_input_ids = input_ids[0]  # Get first sample's input_ids
                        
                        context_before_ids = full_input_ids[context_before_start:first_label_pos]
                        context_after_ids = full_input_ids[last_label_pos + 1:context_after_end]
                        
                        # Show MORE context before and after to understand EOS placement
                        # Show 10 tokens before and after instead of 5
                        context_before_start = max(0, first_label_pos - 40)
                        context_after_end = min(seq_len, last_label_pos + 11)
                        
                        # Get input_ids for context (use the input_ids from function parameters)
                        full_input_ids = input_ids[0]  # Get first sample's input_ids
                        
                        context_before_ids = full_input_ids[context_before_start:first_label_pos]
                        context_after_ids = full_input_ids[last_label_pos + 1:context_after_end]
                        
                        # Show context before
                        if len(context_before_ids) > 0:
                            context_before_text = self.tokenizer.decode(context_before_ids, skip_special_tokens=False)
                            context_before_pred_text = self.tokenizer.decode(context_before_pred, skip_special_tokens=False) if len(context_before_pred) > 0 else ""
                            print(f"\n[TRAINING DEBUG] === CONTEXT BEFORE ANSWER (5 tokens) ===")
                            print(f"[TRAINING DEBUG] Expected: '{context_before_text}'")
                            # print(f"[TRAINING DEBUG] Predicted: '{context_before_pred_text}'")
                        
                        # Decode both expected and predicted answer
                        expected_text = self.tokenizer.decode(expected_tokens, skip_special_tokens=False)
                        predicted_text = self.tokenizer.decode(predicted_tokens, skip_special_tokens=False)
                        
                        print(f"\n[TRAINING DEBUG] === ANSWER TOKENS ===")
                        print(f"[TRAINING DEBUG] Positions [{first_label_pos}:{last_label_pos+1}]")
                        print(f"[TRAINING DEBUG] Expected: '{expected_text}'")
                        print(f"[TRAINING DEBUG] Predicted: '{predicted_text}'")
                        
                        # Show context after
                        if len(context_after_ids) > 0:
                            context_after_text = self.tokenizer.decode(context_after_ids, skip_special_tokens=False)
                            context_after_pred_text = self.tokenizer.decode(context_after_pred, skip_special_tokens=False) if len(context_after_pred) > 0 else ""
                            print(f"\n[TRAINING DEBUG] === CONTEXT AFTER ANSWER (5 tokens) ===")
                            print(f"[TRAINING DEBUG] Expected: '{context_after_text}'")
                            # print(f"[TRAINING DEBUG] Predicted: '{context_after_pred_text}'")
                        
                        print(f"\n[TRAINING DEBUG] === FULL SEQUENCE INFO ===")
                        print(f"[TRAINING DEBUG] Total sequence length: {full_input_ids.shape[0]}")
                        print(f"[TRAINING DEBUG] First label at: {first_label_pos}, Last label at: {last_label_pos}")
                        print()
                    else:
                        print(f"[TRAINING DEBUG] No label positions found to check")
                        
                except Exception as e:
                    print(f"[TRAINING DEBUG] Logit check failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"[TRAINING DEBUG] ========== End Batch Debug ==========\n")
            
            # Stage 2: VLA action prediction with ground truth sensor
            # Check if we need to use all sensors or just dual images
            sensor_input_all = kwargs.get('sensor_input_all', False)

            if sensor_input_all:
                # Use all sensor images (RGB + thermal + radar + acoustic)
                stage2_inputs, stage2_masks = self.prepare_vlm_input_embeddings_all_sensor_images(
                    pixel_values,
                    input_ids,
                    labels,
                    attention_mask,
                    current_state_mask,
                    current_state,
                    fov,
                    debug_training=debug_training,
                )
            else:
                # Use dual images (RGB + single sensor)
                stage2_inputs, stage2_masks = self.prepare_vlm_input_embeddings_two_images(
                    pixel_values,
                    input_ids,
                    labels,
                    attention_mask,
                    current_state_mask,
                    current_state,
                    fov,
                    debug_training=debug_training,
                )
            
            # Forward pass for VLA features
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                vla_outputs = self.model.language_model(
                    past_key_values=None,
                    use_cache=use_cache,
                    output_hidden_states=True,
                    num_logits_to_keep=0,
                    **stage2_inputs
                )
            
            output_hs = vla_outputs.hidden_states[-1]
            
            # Compute action loss
            predicted_actions, action_loss_dict = self._forward_act_model(
                vlm_features=output_hs,
                action_labels=action_labels,
                attention_mask=stage2_masks,
                action_masks=action_masks,
                current_state=current_state,
                current_state_mask=current_state_mask,
                mode=mode,
                repeated_diffusion_steps=self.repeated_diffusion_steps,
                debug_training=debug_training,
            )
            
            # Extract the main loss from the dictionary
            action_loss = action_loss_dict["loss"] if isinstance(action_loss_dict, dict) else action_loss_dict
            
            # Combine losses with weights
            total_loss = vlm_loss_weight * vlm_loss + action_loss_weight * action_loss
            
            # Get training step for conditional debug (from kwargs if available)
            training_step = kwargs.get('training_step', 0)
            
            # Debug: Conditional output based on loss values
            should_debug_loss = False
            debug_reason = ""
            
            # Condition 1: Loss < 0.0001
            if total_loss.item() < 0.0001:
                should_debug_loss = True
                debug_reason = "VERY_LOW_LOSS"

                # # Show ground truth actions - VALID VALUES ONLY (filtered by mask)
                # if action_labels is not None:
                #     gt_action = action_labels[0]  # First sample in batch [T, D]
                #     gt_mask = action_masks[0] if action_masks is not None else None
                #     if gt_mask is not None:
                #         # Filter to get only valid dimensions (where mask > 0)
                #         valid_gt_action = gt_action[0][gt_mask[0] > 0]
                #         num_valid = valid_gt_action.shape[0]
                #         if num_valid != 0:
                #             should_debug_loss  = False
            
            # Condition 2: Step > 10K and loss > 0.1
            if training_step > 10000 and total_loss.item() > 0.2:
                should_debug_loss = True
                debug_reason = "HIGH_LOSS_AFTER_10K"
            
            if should_debug_loss or debug_training:
                print(f"\n[ACTION DEBUG - {debug_reason}] VLM Loss: {vlm_loss.item():.6f}, Action Loss: {action_loss.item():.6f}, Total Loss: {total_loss.item():.6f}")
                
                # Show intermediate action loss components
                if isinstance(action_loss_dict, dict):
                    print(f"[ACTION DEBUG] === Intermediate Action Loss Components ===")
                    for comp_name, comp_loss in action_loss_dict.items():
                        if comp_name != "loss":
                            comp_val = comp_loss.item() if hasattr(comp_loss, 'item') else comp_loss
                            print(f"[ACTION DEBUG]   {comp_name}: {comp_val:.6f}")
                    print(f"[ACTION DEBUG]   total (weighted): {action_loss_dict['loss'].item():.6f}")
                
                # Show action shapes and mask info
                if action_labels is not None:
                    print(f"[ACTION DEBUG] === Action Shapes ===")
                    print(f"[ACTION DEBUG]   action_labels: {action_labels.shape}")
                    print(f"[ACTION DEBUG]   action_masks: {action_masks.shape if action_masks is not None else 'None'}")
                    if action_masks is not None:
                        num_valid_dims = (action_masks[0, 0] > 0).sum().item()
                        num_valid_timesteps = action_masks.shape[1]
                        print(f"[ACTION DEBUG]   valid dims per timestep: {num_valid_dims}, total timesteps: {num_valid_timesteps}")
                
                # Show ground truth actions - ALL VALID TIMESTEPS (filtered by mask)
                if action_labels is not None:
                    gt_action = action_labels[0]  # First sample in batch [T, D]
                    gt_mask = action_masks[0] if action_masks is not None else None
                    num_timesteps = gt_action.shape[0]
                    print(f"[ACTION DEBUG] === GT Actions (all {num_timesteps} timesteps) ===")
                    for t in range(num_timesteps):
                        if gt_mask is not None:
                            valid_gt = gt_action[t][gt_mask[t] > 0]
                            num_valid = valid_gt.shape[0]
                            if num_valid > 0:
                                gt_vals = [f"{v:.4f}" for v in valid_gt.cpu().tolist()]
                                print(f"[ACTION DEBUG]   t={t:2d} ({num_valid} valid dims): [{', '.join(gt_vals)}]")
                            else:
                                print(f"[ACTION DEBUG]   t={t:2d}: (no valid dims)")
                        else:
                            gt_vals = [f"{v:.4f}" for v in gt_action[t].cpu().tolist()]
                            print(f"[ACTION DEBUG]   t={t:2d}: [{', '.join(gt_vals)}]")
                
                # Show predicted actions - ALL VALID TIMESTEPS
                with torch.no_grad():
                    if predicted_actions is not None:
                        pred_action = predicted_actions[0]  # First sample [T, D]
                        num_timesteps = pred_action.shape[0]
                        print(f"[ACTION DEBUG] === Pred Actions (all {num_timesteps} timesteps) ===")
                        for t in range(num_timesteps):
                            if gt_mask is not None:
                                valid_pred = pred_action[t][gt_mask[t] > 0]
                                num_valid = valid_pred.shape[0]
                                if num_valid > 0:
                                    pred_vals = [f"{v:.4f}" for v in valid_pred.cpu().tolist()]
                                    print(f"[ACTION DEBUG]   t={t:2d} ({num_valid} valid dims): [{', '.join(pred_vals)}]")
                                else:
                                    print(f"[ACTION DEBUG]   t={t:2d}: (no valid dims)")
                            else:
                                pred_vals = [f"{v:.4f}" for v in pred_action[t].cpu().tolist()]
                                print(f"[ACTION DEBUG]   t={t:2d}: [{', '.join(pred_vals)}]")
                        
                        # Show per-timestep L2 error between GT and predicted (valid dims only)
                        if action_labels is not None:
                            gt_action = action_labels[0]
                            print(f"[ACTION DEBUG] === Per-Timestep L2 Error (valid dims) ===")
                            for t in range(min(num_timesteps, gt_action.shape[0])):
                                if gt_mask is not None:
                                    valid_mask = gt_mask[t] > 0
                                    if valid_mask.sum() > 0:
                                        gt_t = gt_action[t][valid_mask]
                                        pred_t = pred_action[t][valid_mask]
                                        l2_err = torch.sqrt(((gt_t - pred_t) ** 2).mean()).item()
                                        max_err = (gt_t - pred_t).abs().max().item()
                                        print(f"[ACTION DEBUG]   t={t:2d}: RMSE={l2_err:.6f}, MaxAbsErr={max_err:.6f}")
                                else:
                                    gt_t = gt_action[t]
                                    pred_t = pred_action[t]
                                    l2_err = torch.sqrt(((gt_t - pred_t) ** 2).mean()).item()
                                    max_err = (gt_t - pred_t).abs().max().item()
                                    print(f"[ACTION DEBUG]   t={t:2d}: RMSE={l2_err:.6f}, MaxAbsErr={max_err:.6f}")
                    else:
                        print(f"[ACTION DEBUG] Predicted actions: None (not available in train mode without debug sampling)")
                
                print(f"[ACTION DEBUG] === End Action Debug ===")
                        
            
            # Log loss magnitudes if requested (first 100 steps)
            if log_loss_magnitudes:
                print(f"[VLM-VLA Loss] VLM: {vlm_loss.item():.6f}, Action: {action_loss.item():.6f}, Total: {total_loss.item():.6f}")
            
            loss['vlm_loss'] = vlm_loss
            loss['action_loss'] = action_loss
            loss['total_loss'] = total_loss
            
            return loss

        # Standard VLA mode
        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_values,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache,
            **kwargs,
        )

        _, action_loss = self._forward_act_model(
            vlm_features = output_hs, 
            action_labels = action_labels, 
            attention_mask = inputs_masks, 
            action_masks = action_masks, 
            current_state = current_state, 
            current_state_mask = current_state_mask, 
            mode = mode,
            repeated_diffusion_steps = self.repeated_diffusion_steps,
        )

        self._update_loss(loss, action_loss)

        return loss

    def image_preprocess(self, image: Image, size: Tuple[int, int] = (224, 224)) -> Image:
        width, height = image.size

        return image
        

    def predict_action(
        self, 
        image, 
        instruction: str, 
        current_state, 
        current_state_mask, 
        use_ddim=True, 
        num_ddim_steps=10, 
        cfg_scale=5.0, 
        action_mask_torch=None, 
        fov=None, 
        sample_times=1, 
        use_cache=False,
        task_mode="vla",
        attention_mask=None,
        labels=None,
        train_seg_token: bool = False,
    ) -> np.ndarray:
        """
        Predict action with support for VLA or VLM-VLA co-finetuning modes.
        
        Args:
            image: For vla mode: RGB image (PIL or numpy). For vlm_vla_cofinetune: pixel_values [B, 2, C, H, W]
            instruction: For vla mode: Task instruction string. For vlm_vla_cofinetune: input_ids [B, seq_len]
            current_state: normalized current robot action, [B, D]
            current_state_mask: [B, D]
            use_ddim: Use DDIM sampling
            num_ddim_steps: Number of DDIM steps
            cfg_scale: Classifier-free guidance scale
            action_mask_torch: [B, T, D]
            fov: [B, 2]
            sample_times: Number of samples
            use_cache: Use KV cache
            task_mode: "vla" or "vlm_vla_cofinetune"
            attention_mask: For vlm_vla_cofinetune mode: attention mask [B, seq_len]
            labels: For vlm_vla_cofinetune mode: ground truth labels [B, seq_len]
            
        Returns:
            predicted normalized robot action, [sample_times, T, D]
        """

        B = current_state.shape[0]
        assert B == 1, f"Batch size {B} not supported in predict_action for now."
        
        if task_mode == "vla":
            # Original VLA-only mode
            if isinstance(image, np.ndarray):
                if image.ndim == 3:
                    image = Image.fromarray(image)
                elif image.ndim == 4:
                    image = [Image.fromarray(im) for im in image]
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")
            prefix = '<image>'
            model_inputs = self.processor(text=prefix + instruction, images=image, return_tensors="pt").to('cuda')
            pixel_value = model_inputs['pixel_values']
            input_ids = model_inputs['input_ids']

            # Preprocess Image
            if isinstance(pixel_value, torch.Tensor):
                pixel_value = pixel_value.to('cuda')
            elif isinstance(pixel_value, dict):
                pixel_value = {
                    k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
                }
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
            if pixel_value.dim() == 5:
                pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

            if action_mask_torch is None:
                x_mask = torch.zeros(B, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
                # x_mask[:, :, :102] = 1.0 # predict dual hand actions
                x_mask[:, :, 51:102] = 1.0 # predict right hand actions
                # x_mask[:, :, 0:51] = 1.0 # predict left hand actions
            else:
                x_mask = action_mask_torch.to(input_ids.device)

            attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
            current_state_mask = current_state_mask.to(input_ids.device)
            current_state = current_state.to(input_ids.device)
            fov = fov.to(input_ids.device) if fov is not None else None

            output_hs, inputs_masks = self.prepare_vlm_features(
                pixel_value,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                use_cache=use_cache,
            )
            # handle multiple samples for one input
            samples, _ = self._forward_act_model(
                vlm_features = output_hs,
                attention_mask = inputs_masks,
                action_masks = x_mask,
                current_state = current_state,
                current_state_mask = current_state_mask,
                mode = "eval",
                repeated_diffusion_steps = sample_times,
                cfg_scale = cfg_scale,
                use_ddim = use_ddim,
                num_ddim_steps = num_ddim_steps,
            )
            action_np = samples.cpu().numpy() * x_mask.cpu().numpy()    # sample_times x T x D
            return action_np
            
        elif task_mode == "vlm_vla_cofinetune":
            # VLM-VLA co-finetuning mode with dual or multi-sensor images
            # Follow the same flow as forward function for vlm_vla_cofinetune
            
            # Expecting:
            # - image: pixel_values [1, 2, C, H, W] or [1, 4, C, H, W]
            # - instruction: input_ids [1, seq_len]
            # - labels: labels tensor [1, seq_len]
            # - attention_mask: attention_mask [1, seq_len]
            
            pixel_values = image  # [1, 2/4, C, H, W]
            input_ids = instruction  # Already tokenized [1, seq_len]
            
            # Determine if using all sensors based on pixel_values shape
            sensor_input_all = pixel_values.shape[1] == 4
            
            # Prepare action mask
            if action_mask_torch is None:
                x_mask = torch.zeros(B, self.chunk_size, self.act_model.in_channels, device='cuda')
                x_mask[:, :, 51:102] = 1.0  # predict right hand actions by default
            else:
                x_mask = action_mask_torch.to('cuda')
            
            # Move inputs to cuda
            pixel_values = pixel_values.to('cuda')
            input_ids = input_ids.to('cuda')
            labels = labels.to('cuda')
            if attention_mask is not None:
                attention_mask = attention_mask.to('cuda')
            current_state_mask = current_state_mask.to('cuda')
            current_state = current_state.to('cuda')
            fov = fov.to('cuda') if fov is not None else None
            
            # Stage 2: VLA action prediction with sensor images (matching forward function)
            if sensor_input_all:
                # Use all sensor images (RGB + thermal + radar + acoustic)
                stage2_inputs, stage2_masks = self.prepare_vlm_input_embeddings_all_sensor_images(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    current_state_mask=current_state_mask,
                    current_state=current_state,
                    fov=fov,
                )
            else:
                # Use dual images (RGB + single sensor)
                stage2_inputs, stage2_masks = self.prepare_vlm_input_embeddings_two_images(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    current_state_mask=current_state_mask,
                    current_state=current_state,
                    fov=fov,
                )
            
            # Forward pass for VLA features (matching forward function)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                vla_outputs = self.model.language_model(
                    past_key_values=None,
                    use_cache=use_cache,
                    output_hidden_states=True,
                    num_logits_to_keep=0,
                    **stage2_inputs
                )
            
            output_hs = vla_outputs.hidden_states[-1]
            
            # Compute action predictions (matching forward function)
            samples, _ = self._forward_act_model(
                vlm_features=output_hs,
                attention_mask=stage2_masks,
                action_masks=x_mask,
                current_state=current_state,
                current_state_mask=current_state_mask,
                mode="eval",
                repeated_diffusion_steps=sample_times,
                cfg_scale=cfg_scale,
                use_ddim=use_ddim,
                num_ddim_steps=num_ddim_steps,
            )
            
            action_np = samples.cpu().numpy() * x_mask.cpu().numpy()  # sample_times x T x D
            return action_np
            
        else:
            raise ValueError(f"Unsupported task_mode: {task_mode}")

    def predict_vlm(
        self,
        image,
        instruction: str,
        ground_truth_text: Optional[str] = None,
        ground_truth_sensor: Optional[str] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        train_seg_token: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        use_cache: bool = True,
        debug_training: bool = True,
    ):
        """
        VLM prediction for text generation and optional loss computation.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            instruction: Text instruction
            ground_truth_text: Optional ground truth text for loss computation
            ground_truth_sensor: Optional ground truth sensor token ("none", "thermal", "radar", "audio")
            current_state: Optional current state tensor [B, D]
            current_state_mask: Optional state mask [B, D]
            fov: Optional field of view [B, 2]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repetition)
            no_repeat_ngram_size: Size of n-grams that cannot be repeated (0 = no constraint)
            use_cache: Whether to use KV cache for generation
            
        Returns:
            dict containing:
                - 'predicted_text': Generated text
                - 'predicted_sensor': Predicted sensor token if present
                - 'loss': Loss value if ground truth provided, else None
        """
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        prefix = '<image>Task: '
        
        # Compute loss if ground truth is provided
        loss_value = None
        if ground_truth_text is not None:
            # Prepare ground truth with sensor token if provided
            gt_text = ground_truth_text
            if ground_truth_sensor is not None:
                # Map plain sensor name → special token to match training output format
                _sensor_special_map = {
                    "none": "<None>", "thermal": "<thermal>",
                    "radar": "<radar>", "audio": "<audio>",
                }
                sensor_token = _sensor_special_map.get(
                    ground_truth_sensor.lower(),
                    f"<{ground_truth_sensor}>"  # fallback: wrap in <>
                )
                # Add sensor token at the beginning of ground truth
                gt_text = f"{sensor_token}. {ground_truth_text}"
            
            instruction = instruction.rstrip()
            if not instruction.endswith("."):
                instruction += "."
            # Add the same suffix to instruction as in training dataset
            # full_instruction = instruction + " Question: Choose from none, thermal, radar, audio for task. Tell the target object. Answer:"
            task_desc = "Tell and segment the target object" if train_seg_token else "Tell the target object"
            full_instruction = instruction + f" Question: Choose from <None>, <thermal>, <radar>, <audio> for task. {task_desc}. Answer:"
            
            # IMPORTANT: Disable add_eos_token for prompt to match training data preparation
            original_add_eos_eval = self.tokenizer.add_eos_token
            self.tokenizer.add_eos_token = False
            
            # Tokenize instruction (prompt only, without answer, NO EOS)
            prompt_inputs = self.processor(
                text=prefix + full_instruction,
                images=image,
                return_tensors="pt"
            ).to('cuda')
            
            pixel_values = prompt_inputs['pixel_values']
            prompt_ids = prompt_inputs['input_ids']  # Prompt only - no answer!
            
            if debug_training:
                print(f"\n[DEBUG LOSS] ========== Loss Computation Debug ==========")
                print(f"[DEBUG LOSS] Full instruction: '{full_instruction}'")
                print(f"[DEBUG LOSS] Ground truth text: '{gt_text}'")
            
            # Tokenize ground truth output (answer WITHOUT EOS)
            answer_tokens = self.tokenizer(
                gt_text,
                return_tensors="pt",
                add_special_tokens=False  # Don't add BOS
            ).to('cuda')
            
            answer_ids = answer_tokens['input_ids']  # [B, answer_len]
            
            # Manually append EOS token to match training behavior
            eos_id = self.tokenizer.eos_token_id
            eos_tensor = torch.tensor([[eos_id]], dtype=answer_ids.dtype, device=answer_ids.device)
            answer_ids = torch.cat([answer_ids, eos_tensor], dim=1)
            
            # Restore original tokenizer setting
            self.tokenizer.add_eos_token = original_add_eos_eval

            if debug_training:
                print(f"[DEBUG LOSS] Decoded answer: '{self.tokenizer.decode(answer_ids[0], skip_special_tokens=False)}'")
            
            # ---- CRITICAL: Follow PaddedCollatorForVLM logic exactly ----
            # Match the collator: input_ids = [prompt | answer], labels = [-100 | answer]
            # token_type_ids = [0 for prompt | 1 for answer] (prefix-LM causal masking)
            prompt_len = prompt_ids.shape[1]
            answer_len = answer_ids.shape[1]
            total_len = prompt_len + answer_len
            
            pad_token_id = self.tokenizer.pad_token_id
            
            # Create input_ids: [prompt tokens | answer tokens] (causal teacher forcing)
            input_ids = torch.full((1, total_len), pad_token_id, dtype=prompt_ids.dtype, device=prompt_ids.device)
            input_ids[0, :prompt_len] = prompt_ids[0]
            input_ids[0, prompt_len:prompt_len+answer_len] = answer_ids[0]
            
            # Create labels: [-100 for prompt positions | answer tokens]
            labels = torch.full((1, total_len), -100, dtype=answer_ids.dtype, device=answer_ids.device)
            labels[0, prompt_len:prompt_len+answer_len] = answer_ids[0]
            
            # Create token_type_ids: 0 for prefix (prompt), 1 for suffix (answer)
            token_type_ids = torch.ones((1, total_len), dtype=torch.long, device=prompt_ids.device)
            token_type_ids[0, :prompt_len] = 0
            
            # Truncate if needed
            model_max_length = getattr(self.tokenizer, 'model_max_length', None)
            if model_max_length is None or model_max_length > 100000:
                model_max_length = 2048  # Reasonable default
            
            if total_len > model_max_length:
                input_ids = input_ids[:, :model_max_length]
                labels = labels[:, :model_max_length]
                token_type_ids = token_type_ids[:, :model_max_length]
            
            # Attention mask: 1 for prompt + answer positions
            attention_mask = input_ids.ne(pad_token_id).long()
            
            # Show which tokens are being predicted
            valid_label_mask = labels[0] != -100
            valid_label_ids = labels[0][valid_label_mask]
            
            # Preprocess pixel values
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to('cuda')
            elif isinstance(pixel_values, dict):
                pixel_values = {
                    k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to('cuda') 
                    for k in pixel_values[0]
                }
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
            
            if pixel_values.dim() == 5:
                pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            
            # Prepare optional inputs
            if current_state is not None:
                current_state = current_state.to(input_ids.device)
            if current_state_mask is not None:
                current_state_mask = current_state_mask.to(input_ids.device)
            if fov is not None:
                fov = fov.to(input_ids.device)
            
            # Compute loss using forward pass
            if debug_training:
                print(f"[DEBUG LOSS] Calling forward pass for loss computation...")
            with torch.no_grad():
                loss_dict = self.forward(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    token_type_ids=token_type_ids,
                    current_state=current_state,
                    current_state_mask=current_state_mask,
                    fov=fov,
                    use_cache=False,
                    mode="train",
                    task_mode="vlm_finetune",
                    debug_training=False,
                )
            
            loss_value = loss_dict.get('vlm_loss', None)
            if loss_value is not None:
                loss_value = loss_value.item()
                if debug_training:
                    print(f"[DEBUG LOSS] Computed loss: {loss_value:.4f}")
            if debug_training:
                print(f"[DEBUG LOSS] ========== End Loss Computation Debug ==========\n")
        
        # Generate prediction using the same instruction format as training
        # full_instruction = "Here is instruction of robot manipulation task: " + instruction + " Choose a sensor from none, thermal, radar, acoustic, and also tell the target object."
        instruction = instruction.rstrip()
        if not instruction.endswith("."):
            instruction += "."
        # full_instruction = instruction + " Question: Choose from none, thermal, radar, audio for task. Tell the target object. Answer:"
        task_desc = "Tell and segment the target object" if train_seg_token else "Tell the target object"
        full_instruction = instruction + f" Question: Choose from <None>, <thermal>, <radar>, <audio> for task. {task_desc}. Answer:"
        
        # Temporarily disable add_eos_token for generation input
        original_add_eos = self.tokenizer.add_eos_token
        self.tokenizer.add_eos_token = False
        
        model_inputs = self.processor(
            text=prefix + full_instruction,
            images=image,
            return_tensors="pt"
        ).to('cuda')
        
        # Restore original setting
        self.tokenizer.add_eos_token = original_add_eos
        
        pixel_values = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        # Preprocess pixel values
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to('cuda')
        elif isinstance(pixel_values, dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to('cuda') 
                for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
        
        # Prepare optional inputs
        if current_state is not None:
            current_state = current_state.to(input_ids.device)
        if current_state_mask is not None:
            current_state_mask = current_state_mask.to(input_ids.device)
        if fov is not None:
            fov = fov.to(input_ids.device)
        
        # Autoregressive generation using model.generate()
        # This matches causal teacher-forcing training: each token conditions
        # on all previous tokens via causal attention.
        # NOTE: use_cache=False avoids Gemma2 HybridCache device mismatch
        # when model is loaded with map_location="cpu" then moved to CUDA.
        with torch.no_grad():
            generated_output = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                use_cache=False,
            )

        # Strip prompt tokens to get only generated tokens
        generated_ids = generated_output[0, input_ids.shape[1]:]
        
        # Decode generated tokens
        if debug_training:
            print(f"[DEBUG] Length of generated text: {len(generated_ids)}")
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        raw_generated_text = generated_text  # Save raw text before any cleaning
        if debug_training:
            print(f"[DEBUG] Raw generated text before cleaning: '{generated_text}'")
        
        # Extract predicted sensor token if present (do this before cleaning)
        predicted_sensor = None
        import re

        # sensor_tokens = ["none", "thermal", "radar", "audio"]
        # for token in sensor_tokens:
        #     if re.search(rf"\b{token}\b", generated_text, flags=re.IGNORECASE):
        #         predicted_sensor = token
        #         break
        # for token in sensor_tokens:
        #     generated_text = re.sub(token, "", generated_text, flags=re.IGNORECASE)

        
        sensor_tokens = ["<thermal>", "<radar>", "<audio>", "<None>"]
        earliest_position = float('inf')
        
        for token in sensor_tokens:
            # escape <> properly
            pattern = re.escape(token)
            match = re.search(pattern, generated_text, flags=re.IGNORECASE)
            if match and match.start() < earliest_position:
                earliest_position = match.start()
                # strip <> and normalize
                predicted_sensor = token.strip("<>").lower()
        
        # Remove sensor tokens from text
        for token in sensor_tokens:
            generated_text = re.sub(
                re.escape(token), "", generated_text, flags=re.IGNORECASE
            )

        # Filter out loc and seg tokens from object text
        generated_text = re.sub(r'<loc\d{4}>', '', generated_text)
        generated_text = re.sub(r'<seg\d{3}>', '', generated_text)

        # Normalize whitespace
        generated_text = re.sub(r"\s+", " ", generated_text).strip()


        # Clean up generated text (remove other special tokens)
        # Remove other standard special tokens (BOS, EOS, PAD)
        for token in ['<bos>', '<eos>', '<pad>', '<unk>', '.']:
            generated_text = generated_text.replace(token, '').strip()
        if debug_training:
            print(f"[DEBUG] Cleaned generated text: '{generated_text}'")

        # seen = set()
        # deduped_words = []
        # for w in generated_text.split():
        #     wl = w.lower()  # case-insensitive deduplication
        #     if wl not in seen:
        #         seen.add(wl)
        #         deduped_words.append(w)
        # generated_text = " ".join(deduped_words)
        # if debug_training:
        #     print(f"[DEBUG] Deduplicated generated text: '{generated_text}'")

        # # SpaCy-based object extraction (commented out - using raw text directly)
        # COLOR_WORDS = {
        #     # basic
        #     "black", "white", "gray", "grey",
        #     "red", "orange", "yellow", "green",
        #     "blue", "purple", "pink", "brown",
        #     "beige",
        #     # shades
        #     "light", "dark",
        #     "bright", "pale",
        #     "deep", "clear"
        # }
        # import spacy
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(generated_text)
        # words = []
        # for token in doc:
        #     # skip spaces / punctuation
        #     if token.is_space or token.is_punct:
        #         continue
        #     # keep ASCII English words only
        #     if not re.fullmatch(r"[A-Za-z]+", token.text):
        #         continue
        #     # Skip capitalized Proper Nouns (hallucinated symbols/classes)
        #     if token.pos_ == "PROPN" and token.text[0].isupper():
        #         continue
        #     t = token.text.lower()
        #     # keep adjectives (colors, patterns like "striped", etc.) and continue
        #     if token.pos_ == "ADJ" or t in COLOR_WORDS:
        #         words.append(t)
        #         continue
        #     # first noun or non-adjective word → include it and stop
        #     words.append(t)
        #     break
        # generated_text = " ".join(words)
        # if debug_training:
        #     print(f"[DEBUG] Generated text after filtering: {generated_text}")
        
        # generated_text = " ".join(generated_text.split()[:3])


        if debug_training:
            print(f"[DEBUG] Final generated text: '{generated_text}'")

        
        return {
            'predicted_text': generated_text,
            'raw_generated_text': raw_generated_text,
            'predicted_sensor': predicted_sensor,
            'loss': loss_value,
        }

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """

        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss