import lightning as L
import torch
from peft import LoraConfig, get_peft_model_state_dict
import prodigyopt
from typing import Dict, List, Optional, Union
import numpy as np

# Import local modules
from ..pipeline_flux_omini_kontext import FluxOminiKontextPipeline
from ..pipeline_tools import encode_images, prepare_text_input


class FluxOminiKontextModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux Omini Kontext pipeline
        self.flux_pipe: FluxOminiKontextPipeline = (
            FluxOminiKontextPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline components
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        # Initialize joint attention kwargs
        self.joint_attention_kwargs = {}

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            self.flux_pipe.load_lora_weights(lora_path, adapter_name="default")
            # Get trainable parameters (LoRA layers)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # Get trainable parameters (LoRA layers)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxOminiKontextPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        # Extract inputs from batch
        input_images = batch["input_image"]  # Main input image
        reference_images = batch["reference_image"]  # Reference image
        target_images = batch["target_image"]  # Target image
        prompts = batch["prompt"]  # Text prompt
        reference_deltas = batch.get("reference_delta", [[0, 0, 0]])  # Position delta for reference

        # Prepare inputs
        with torch.no_grad():
            # Prepare input image
            x_0, x_ids = encode_images(self.flux_pipe, target_images)
          
            x_init, init_img_ids = encode_images(self.flux_pipe, input_images)
            # Prepare reference image with delta
            x_ref, ref_img_ids = encode_images(self.flux_pipe, reference_images)
            
            # Apply position delta to reference image IDs
            delta = reference_deltas[0]
            ref_img_ids[:, 0] += delta[0]
            ref_img_ids[:, 1] += delta[1]
            ref_img_ids[:, 2] += delta[2]

            # Combine input and reference images
            condition = torch.cat([x_init, x_ref], dim=1)
            condition_ids = torch.cat([init_img_ids, ref_img_ids], dim=0)

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )

            # Prepare t and x_t
            t = torch.rand((input_images.shape[0],), device=self.device)
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            latent_model_input = torch.cat([x_t, condition], dim=1)
            latent_ids = torch.cat([x_ids, condition_ids], dim=0)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
        pred = pred[:, : x_t.size(1)]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss

    def validation_step(self, batch, batch_idx):
        # Similar to training step but for validation
        with torch.no_grad():
            loss = self.step(batch)
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Log training loss at the end of each epoch
        self.log("train_loss", self.log_loss, prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        # Save LoRA weights when checkpoint is saved
        if hasattr(self, 'save_lora'):
            # You can implement custom saving logic here
            pass

    def forward(self, batch):
        # Forward pass for inference
        return self.step(batch)

