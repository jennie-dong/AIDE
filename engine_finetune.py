# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, 
    accuracy_score
)
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(data_loader, model, device, use_amp=False):
    import os, cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.special import softmax
    from sklearn.metrics import accuracy_score, average_precision_score

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    save_root = "eval_cam_results_db_bowen"
    os.makedirs(save_root, exist_ok=True)

    # ====== AIDE: hook 到 model_min / model_max 的 layer4 ======
    acts = {"min": [], "max": []}
    grads = {"min": [], "max": []}

    def fwd_hook_min(m, inp, out):
        acts["min"].append(out)

    def bwd_hook_min(m, gin, gout):
        grads["min"].append(gout[0])

    def fwd_hook_max(m, inp, out):
        acts["max"].append(out)

    def bwd_hook_max(m, gin, gout):
        grads["max"].append(gout[0])

    h_f_min = model.model_min.layer4.register_forward_hook(fwd_hook_min)
    h_b_min = model.model_min.layer4.register_full_backward_hook(bwd_hook_min)
    h_f_max = model.model_max.layer4.register_forward_hook(fwd_hook_max)
    h_b_max = model.model_max.layer4.register_full_backward_hook(bwd_hook_max)

    for index, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0].to(device, non_blocking=True)   # [B,5,3,H,W]
        target = batch[1].to(device, non_blocking=True)
        image_paths = batch[2]                            # list[str]
        patch_info = batch[3]                            # list[dict]

        B, V, _, H, W = images.shape

        # 每个 batch 前清空 list（否则会把上一个 batch 的也混进来）
        acts["min"].clear(); acts["max"].clear()
        grads["min"].clear(); grads["max"].clear()

        with torch.enable_grad():
            output = model(images)
            if isinstance(output, dict):
                output = output["logits"]
            loss = criterion(output, target)

            # 对 fake 类做 backward（假设 index=1 是 fake）
            score = output[:, 1].sum()
            model.zero_grad(set_to_none=True)
            score.backward()

            # ====== 现在 hooks 已经抓到 4 次调用的 layer4 特征/梯度 ======
            # AIDE forward 调用顺序：
            # model_min: 第0次 -> x_minmin(view0), 第1次 -> x_minmin1(view2)
            # model_max: 第0次 -> x_maxmax(view1), 第1次 -> x_maxmax1(view3)

            # 你要看哪个“视图/分支”的 CAM？
            # 例如：看 view1 (x_maxmax) -> model_max 第0次
            branch = "max"
            call_idx = 0          # 0: x_maxmax(view1), 1: x_maxmax1(view3)

            feat = acts[branch][call_idx]   # [B,C,Hc,Wc]
            g = grads[branch][call_idx]     # [B,C,Hc,Wc]

            # Grad-CAM
            weights = g.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * feat).sum(dim=1))   # [B,Hc,Wc]

            # resize 到输入图大小
            cam = F.interpolate(cam.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        # ====== 对 batch 中每张图保存（画框 + heatmap overlay） ======
        for b in range(B):
            cam_np = cam[b].detach().float().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

            # bbox: 阈值 + 最大连通域
            binary = (cam_np > 0.5).astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
            bbox = None
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_id = np.argmax(areas) + 1
                x = stats[max_id, cv2.CC_STAT_LEFT]
                y = stats[max_id, cv2.CC_STAT_TOP]
                w = stats[max_id, cv2.CC_STAT_WIDTH]
                h = stats[max_id, cv2.CC_STAT_HEIGHT]
                bbox = (x, y, x + w, y + h)

            # 取要可视化的那张 3 通道图（view_id）
            img_tensor = images[b, -1]  # [3,H,W]

            # 反归一化（如果你 transform_train 用的是 ImageNet mean/std）
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
            img_vis = (img_tensor * std + mean).clamp(0, 1)
            img_vis = (img_vis.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_vis, 0.6, heatmap, 0.4, 0)

            if bbox is not None:
                cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)

            base = os.path.basename(image_paths[b])
            name = os.path.splitext(base)[0]
            out_path = os.path.join(save_root, f"{name}_{branch}{call_idx}.png")
            cv2.imwrite(out_path, overlay[..., ::-1])  # RGB->BGR
            out_path = os.path.join(save_root, f"original_{name}_{branch}{call_idx}.png")
            cv2.imwrite(out_path, img_vis[..., ::-1])

            patch_indices = {
                "minmin": (patch_info["minmin"][0][b], patch_info["minmin"][1][b], patch_info["minmin"][2][b], patch_info["minmin"][3][b]),
                "maxmax": (patch_info["maxmax"][0][b], patch_info["maxmax"][1][b], patch_info["maxmax"][2][b], patch_info["maxmax"][3][b]),
                "minmin1": (patch_info["minmin1"][0][b], patch_info["minmin1"][1][b], patch_info["minmin1"][2][b], patch_info["minmin1"][3][b]),
                "maxmax1": (patch_info["maxmax1"][0][b], patch_info["maxmax1"][1][b], patch_info["maxmax1"][2][b], patch_info["maxmax1"][3][b]),
            }

            # 颜色区分（BGR for OpenCV, but we draw on RGB then convert）
            colors = {
                "minmin":  (0, 255, 0),    # 绿色
                "maxmax":  (255, 0, 0),    # 蓝色
                "minmin1": (255, 255, 0),  # 青色
                "maxmax1": (0, 0, 255),    # 红色
            }

            H_vis, W_vis = img_vis.shape[:2]

            patch_vis_all = img_vis.copy()

            for name_patch, pidx in patch_indices.items():                
                x0n, y0n, x1n, y1n = pidx
                
                x0 = int(x0n * W_vis)
                y0 = int(y0n * H_vis)
                x1 = int(x1n * W_vis)
                y1 = int(y1n * H_vis)

                cv2.rectangle(
                    patch_vis_all,
                    (x0, y0),
                    (x1, y1),
                    colors[name_patch],
                    2
                )

                # 可选：标文字
                cv2.putText(
                    patch_vis_all,
                    name_patch,
                    (x0 + 2, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors[name_patch],
                    1,
                    cv2.LINE_AA
                )

            # 保存
            patch_out_path = os.path.join(
                save_root,
                f"patch_all_{name}.png"
            )
            cv2.imwrite(patch_out_path, patch_vis_all[..., ::-1])  # RGB -> BGR

            
        # ====== 下面保持你原来的 metric 统计即可（略） ======
        with torch.no_grad():
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 2))
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=B)
        metric_logger.meters["acc5"].update(acc5.item(), n=B)

    # 清理 hooks
    h_f_min.remove(); h_b_min.remove()
    h_f_max.remove(); h_b_max.remove()

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # 你的 ap/acc 统计若需要可接回去（这里省略）
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 0, 0
