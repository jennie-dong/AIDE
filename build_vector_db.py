import torch
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import multiprocessing

from data.datasets import TestDataset
import models.AIDE as AIDE

def custom_collate(batch):
    """
    自定义的数据整理函数，用于处理 (tensor, label, path) 格式的 batch
    让 DataLoader 能够支持 batch_size > 1
    """
    # batch 是一个列表，每个元素是 dataset[i] 的返回值: (tensors, label, path)
    
    # 1. 堆叠图像数据 [B, 5, C, H, W]
    tensors = torch.stack([item[0] for item in batch])
    
    # 2. 堆叠标签数据 [B]
    labels = torch.tensor([item[1] for item in batch])
    
    # 3. 收集路径 (保持为列表字符串)
    paths = [item[2] for item in batch]
    
    return tensors, labels, paths

def build_database(args):
    # 1. 检查 GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用显卡: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")

    # 2. 加载模型
    print(f"Loading model from {args.resume}...")
    model = AIDE.AIDE(resnet_path=None, convnext_path=None)
    
    # 加载权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    # 忽略 "No pretrained weights"，因为马上覆盖
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()

    # 3. 加载数据
    dataset_val = TestDataset(is_train=False, args=args)
    data_loader = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=custom_collate
    )

    # 4. 提取特征库
    vector_db = {} 
    print(f"Start building vector database with Batch Size {args.batch_size}...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # 获取 batch 数据
            images = batch[0].to(device) # [B, 5, C, H, W]
            labels = batch[1].to(device) # [B]
            paths = batch[2]             # List of strings

            # 模型推理
            # 使用混合精度 (autocast) 进一步加速
            with torch.cuda.amp.autocast():
                output = model(images)
                logits = output['logits']
                features = output['feature'] # [B, 256]

            # 获取预测结果
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # 遍历当前 Batch 中的每一张图
            batch_len = labels.shape[0]
            for i in range(batch_len):
                # 模型预测为 Fake (1) 且 真实标签为 Fake (1)
                if preds[i].item() == 1 and labels[i].item() == 1:
                    # 归一化特征
                    feat_vec = features[i]
                    norm_feat = F.normalize(feat_vec.unsqueeze(0), p=2, dim=1).cpu()
                    
                    # 获取文件名
                    filename = os.path.basename(paths[i])
                    
                    # 存入字典
                    # squeeze(0) 存成 [256] 而不是 [1, 256]，方便后续处理
                    vector_db[filename] = norm_feat

    # 5. 保存
    save_path = os.path.join(args.output_dir, 'fake_image_vectors.pt')
    torch.save(vector_db, save_path)
    print(f"Database built! Saved {len(vector_db)} vectors to {save_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to test images')
    parser.add_argument('--resume', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output_dir', type=str, default='./', help='Where to save the .pt database')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for GPU acceleration') # 新增参数
    
    # 兼容参数
    parser.add_argument('--data_path', type=str, default='') 
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    build_database(args)