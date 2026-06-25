import torch
def sanity_check(
        self_supervised_train_loader,
        use_data_parallel,
        self_supervised_model,
        device
):
    first_batch = next(iter(self_supervised_train_loader))

    if use_data_parallel:
        self_supervised_model = torch.nn.DataParallel(self_supervised_model)

    model_for_count = self_supervised_model.module if isinstance(self_supervised_model, torch.nn.DataParallel) else self_supervised_model

    total_params = sum(
        parameter.numel() for parameter in model_for_count.parameters()
    )
    trainable_params = sum(parameter.numel() for parameter in model_for_count.parameters() if parameter.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    self_supervised_model.eval()
    with torch.no_grad():
        image = first_batch['image'][:1].to(device)
        patches = first_batch['patches'][:1].to(device)
        reconstructed_image, patch_attn_maps, foreground_ratio, foreground_prior, patch_scores, patch_weights = self_supervised_model(
            image,
            patches
        )

    print(f'Input image shape: {tuple(image.shape)}')
    print(f'Input patches shape: {tuple(patches.shape)}')
    print(f'Reconstructed image shape: {tuple(reconstructed_image.shape)}')
    print(f'Foreground ratio shape: {tuple(foreground_ratio.shape)}')
    print(f'Patch weights shape: {tuple(patch_weights.shape)}')
    print(f'Number of attention maps: {len(patch_attn_maps)}')
    print(f'First attention map shape: {tuple(patch_attn_maps[0].shape)}')