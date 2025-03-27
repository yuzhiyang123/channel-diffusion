import os
from accelerate import Accelerator
import torch
from tqdm.auto import tqdm

MSEComplex = lambda x: torch.mean(x.real * x.real + x.imag * x.imag)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader,
               test_dataloader, lr_scheduler, data_preprocess):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.exp_name, "results", "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(os.path.join(config.exp_name, "results"), exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        #progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        #progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_channels = data_preprocess(batch)
            with torch.no_grad():
                mean = torch.mean(clean_channels, dim=(1, 2), keepdim=True)
                var1 = torch.mean(clean_channels.real * clean_channels.real, dim=(1, 2), keepdim=True) - mean.real * mean.real
                var2 = torch.mean(clean_channels.imag * clean_channels.imag, dim=(1, 2), keepdim=True) - mean.imag * mean.imag
                std = torch.sqrt(var1 + var2)
            clean_channels = (clean_channels - mean) / std

            # Sample noise to add to the images
            noise = torch.randn_like(clean_channels)
            bs = clean_channels.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_channels.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_channels = noise_scheduler.add_noise(clean_channels, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                pred = model(noisy_channels, timesteps)
                if config.prediction_type == "epsilon":
                    loss = MSEComplex(pred - noise)
                elif config.prediction_type == "sample":
                    loss = MSEComplex(pred - clean_channels)
                elif config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            #progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            #progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        lr_scheduler.step()
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            if True:
                loss = test(model, test_dataloader, noise_scheduler, data_preprocess)
                logs = {"test loss": loss}
                accelerator.log(logs, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save(model.state_dict(),
                           os.path.join(config.exp_name, "results", "model_epoch%d.pth" % (epoch+1)))


# Used for testing or evaluation
def test(model, test_dataloader, noise_scheduler, data_preprocess):
    loss = 0.
    n_data = 0
    for step, batch in enumerate(test_dataloader):
        clean_channels = data_preprocess(batch)
        with torch.no_grad():
            mean = torch.mean(clean_channels, dim=(1, 2), keepdim=True)
            var1 = torch.mean(clean_channels.real * clean_channels.real, dim=(1, 2), keepdim=True) - mean.real * mean.real
            var2 = torch.mean(clean_channels.imag * clean_channels.imag, dim=(1, 2), keepdim=True) - mean.imag * mean.imag
            std = torch.sqrt(var1 + var2)
        clean_channels = (clean_channels - mean) / std
        # Sample noise to add to the images
        noise = torch.randn_like(clean_channels)
        bs = clean_channels.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_channels.device,
            dtype=torch.int64
        )
        with torch.no_grad():
            noisy_channels = noise_scheduler.add_noise(clean_channels, noise, timesteps)
            # Predict the noise residual
            pred = model(noisy_channels, timesteps)
            loss0 = MSEComplex(pred - clean_channels)
        loss += loss0.item()
        n_data += bs
    return loss / n_data

def train_loop_benchmark(config, model, noise_scheduler, optimizer, train_dataloader,
               test_dataloader, lr_scheduler, data_preprocess):
    pass
