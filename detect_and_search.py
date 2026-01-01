import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os

from torchvision import transforms
from data.dct import DCT_base_Rec_Module
import models.AIDE as AIDE
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# 定义推理Transform
transform_before_test = transforms.Compose([transforms.ToTensor()])
transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def process_single_image(image_path, dct_module):
    """将单张图片处理成模型需要的 Tensor 格式"""
    image = Image.open(image_path).convert('RGB')
    
    # 1. DCT 前处理
    img_tensor = transform_before_test(image) 
    
    # DCT 接收 Tensor 输入
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_tensor)

    # 2. 归一化和 Resize
    x_0 = transform_train(img_tensor) # [C, 256, 256]
    
    # DCT 输出是 [1, C, 256, 256]，需要 squeeze 掉 batch 维度才能进行 transform
    x_minmin = transform_train(x_minmin.squeeze(0))
    x_maxmax = transform_train(x_maxmax.squeeze(0))
    x_minmin1 = transform_train(x_minmin1.squeeze(0))
    x_maxmax1 = transform_train(x_maxmax1.squeeze(0))

    # Stack [5, C, H, W] -> Add Batch Dim -> [1, 5, C, H, W]
    input_tensor = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0).unsqueeze(0)
    return input_tensor

def main(args):
    device = torch.device(args.device)
    
    # 1. 加载数据库
    print(f"Loading vector database from {args.db_path}...")
    vector_db = torch.load(args.db_path)
    
    # 2. 加载模型
    print("Loading model...")
    model = AIDE.AIDE(resnet_path=None, convnext_path=None)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['model']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 初始化 DCT 
    dct = DCT_base_Rec_Module().to('cpu') # DCT 处理通常在 CPU 上做或者需要适配 device

    # 3. 处理输入图片
    print(f"Processing image: {args.image_path}")
    input_tensor = process_single_image(args.image_path, dct)
    input_tensor = input_tensor.to(device)

    # 4. 推理
    with torch.no_grad():
        output = model(input_tensor)
        logits = output['logits']
        feature = output['feature'] # [1, 256]
        
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][1].item() # Fake 的概率

    print(f"\nResult:")
    print(f"Prediction: {'FAKE (AI-Generated)' if pred_label == 1 else 'REAL'}")
    print(f"Confidence (Fake): {confidence:.4f}")

    # 5. 如果是 Fake，进行检索
    if pred_label == 1:
        print("\nSearching for similar images in database...")
        
        # 归一化查询向量
        query_feat = F.normalize(feature, p=2, dim=1) # [1, 256]
        
        similarities = []
        for filename, db_feat in vector_db.items():
            db_feat = db_feat.to(device) # [1, 256]
            
            # 计算余弦相似度
            sim = F.cosine_similarity(query_feat, db_feat).item()
            similarities.append((filename, sim))
        
        # 排序取 Top 3
        # reverse=True 表示从大到小 (相似度 1.0 最高)
        top3 = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        
        print("\nTop 3 Similar AI Images:")
        for rank, (fname, score) in enumerate(top3, 1):
            print(f"{rank}. Filename: {fname} | Similarity: {score:.4f}")
    else:
        print("\nImage classified as Real. Skipping similarity search.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--resume', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--db_path', type=str, default='fake_image_vectors.pt', help='Path to database file')
    parser.add_argument('--device', default='cuda', help='device')
    
    args = parser.parse_args()
    main(args)