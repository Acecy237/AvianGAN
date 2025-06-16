# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True, config={}):
    param = {}
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            ys = []
            for f in AUGMENT_FNS[p]:
                x, y = f(x, config)
                ys.append(y)
            ys = torch.cat(ys, 1)
            param[p] = ys
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x, param


def rand_brightness(x, config):
    ratio = config.get('brightness', 1.0)
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x = x + (r - 0.5) * ratio
    return x, r.view(-1, 1)


def rand_saturation(x, config):
    ratio = config.get('saturation', 1.0)
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * ((r * 2 - 1) * ratio + 1) + x_mean
    return x, r.view(-1, 1)


def rand_contrast(x, config):
    ratio = config.get('contrast', 1.0)
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * ((r - 0.5) * ratio + 1) + x_mean
    return x, r.view(-1, 1)


def rand_translation(x, config):
    ratio = config.get('translation', 0.125)
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x, torch.cat([translation_x.view(-1, 1).float() / shift_x / 2.0 + 0.5,
                         translation_y.view(-1, 1).float() / shift_y / 2.0 + 0.5], 1)


def rand_cutout(x, config):
    ratio = config.get('cutout', 0.5)
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, torch.cat([offset_x.view(-1, 1).float() / (x.size(2) - cutout_size[0] % 2),
                         offset_y.view(-1, 1).float() / (x.size(3) - cutout_size[1] % 2)], 1)


def rand_log_amplitude(x, config):
    """
    对数幅度缩放，用于模拟频谱图的动态范围调整。
    Args:
        x: 输入频谱图 (batch_size, 1, height, width)
        config: 配置字典，包含缩放强度
    Returns:
        增强后的图像和增强参数
    """
    ratio = config.get('log_amplitude', 0.2)  # 缩放强度
    r = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)  # 随机缩放参数
    x = x * (1 + (r - 0.5) * ratio)  # 缩放增强
    return x, r.view(-1, 1)


def rand_time_shift(x, config):
    """
    时间轴平移，用于模拟信号的时间偏移。
    Args:
        x: 输入频谱图 (batch_size, 1, height, width)
        config: 配置字典，包含平移比例
    Returns:
        增强后的图像和归一化偏移参数
    """
    ratio = config.get('time_shift', 0.1)  # 平移比例
    shift = int(x.size(-1) * ratio)  # 平移的像素数
    offset = torch.randint(-shift, shift + 1, size=(x.size(0),), device=x.device)  # 随机偏移量
    for i in range(x.size(0)):
        x[i] = torch.roll(x[i], shifts=offset[i].item(), dims=-1)  # 按时间轴滚动
    return x, offset.view(-1, 1).float() / shift


def rand_frequency_mask(x, config):
    """
    频率掩码，用于模拟频谱图的频率丢失。
    Args:
        x: 输入频谱图 (batch_size, 1, height, width)
        config: 配置字典，包含掩码比例
    Returns:
        增强后的图像和掩码参数
    """
    ratio = config.get('freq_mask', 0.2)  # 掩码比例
    mask_size = int(x.size(-2) * ratio)  # 掩码大小
    offset = torch.randint(0, x.size(-2) - mask_size + 1, size=(x.size(0),), device=x.device)  # 随机掩码起始点
    mask = torch.ones_like(x)
    for i in range(x.size(0)):
        mask[i, :, offset[i]:offset[i] + mask_size, :] = 0  # 设置掩码区域
    x = x * mask
    return x, offset.view(-1, 1).float() / x.size(-2)


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'log_amplitude': [rand_log_amplitude],
    'time_shift': [rand_time_shift],
    'freq_mask': [rand_frequency_mask]
}

AUGMENT_DIM = {
    'color': 3,
    'translation': 2,
    'cutout': 2,
    'log_amplitude': 1,
    'time_shift': 1,
    'freq_mask': 1
}
