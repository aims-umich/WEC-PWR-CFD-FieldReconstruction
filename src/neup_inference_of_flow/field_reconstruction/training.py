import torch
import torch.nn as nn



def masked_mse_2d(pred, target, miss_mask, eps=1e-8):
    num = ((pred - target) ** 2 * miss_mask).sum()
    den = miss_mask.sum().clamp_min(eps)
    return num / den



def masked_mse_multilevel(pred, target, miss_mask, eps=1e-8):
    num = ((pred - target) ** 2 * miss_mask).sum(dim=(1, 2, 3, 4))
    den = miss_mask.sum(dim=(1, 2, 3, 4)).clamp_min(eps)
    return (num / den).mean()



def masked_mse_3d(pred, target, miss_mask, eps=1e-8):
    num = ((pred - target) ** 2 * miss_mask).sum()
    den = miss_mask.sum().clamp_min(eps)
    return num / den



def default_batch_adapter(batch, device):
    x, y, m_miss, m_obs, m_geom = batch
    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        m_miss.to(device, non_blocking=True),
        m_obs.to(device, non_blocking=True),
        m_geom.to(device, non_blocking=True),
    )



def build_optimizer_scheduler_scaler(model, hp, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hp["lr_factor"],
        patience=hp["patience"],
        verbose=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=hp["use_amp"] and device.type == "cuda")
    return optimizer, scheduler, scaler



def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    train_mode,
    loss_fn,
    grad_clip=None,
    amp_enabled=False,
    batch_adapter=None,
):
    model.train(mode=train_mode)
    adapter = batch_adapter or default_batch_adapter
    use_amp = amp_enabled and device.type == "cuda" and scaler is not None

    total = 0.0
    n = 0
    for batch in loader:
        x, y, m_miss, _, _ = adapter(batch, device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_hat = model(x)
                loss = loss_fn(y_hat, y, m_miss)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        else:
            with torch.no_grad():
                y_hat = model(x)
                loss = loss_fn(y_hat, y, m_miss)

        batch_size = x.size(0)
        total += loss.item() * batch_size
        n += batch_size

    return total / max(n, 1)
