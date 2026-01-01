import torch
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import open_clip  # 核心依赖：直接调用 OpenCLIP
from PIL import Image
import glob

def build_database(args):
    # 1. 检查设备
    # 优先使用 GPU，如果没有则回退到 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 [Test Script] Building Semantic DB on {device}...")

    # 2. 加载 OpenCLIP 模型 (ViT-B-32)
    # 注意：必须与 explainable_api.py 中使用的模型完全一致，保证特征空间对齐
    print("🔄 Loading OpenCLIP model (ViT-B-32)...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请确保已安装 open_clip_torch: pip install open_clip_torch")
        return

    # 3. 扫描图片
    # 支持递归搜索所有常见图片格式
    print(f"📂 Scanning images in: {args.data_path}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        # 递归查找 (例如 D:\...\**\*.jpg)
        found = glob.glob(os.path.join(args.data_path, "**", ext), recursive=True)
        image_files.extend(found)
    
    # 过滤逻辑（可选）：只处理文件名或路径包含 'fake' 的图片
    # 如果你的文件夹里混有真图，建议取消下面这行的注释
    # image_files = [f for f in image_files if "fake" in f.lower() or "1_fake" in f.lower()]

    if not image_files:
        print(f"❌ Error: 未找到任何图片! 请检查路径: {args.data_path}")
        return

    print(f"✅ Found {len(image_files)} images. Start extracting features...")

    # 4. 提取特征并构建数据库
    vector_db = {}
    success_count = 0
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Extracting"):
            try:
                # A. 预处理
                image = Image.open(img_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # B. 提取特征
                features = model.encode_image(image_tensor)
                
                # C. 归一化 (关键步骤：余弦相似度要求向量归一化)
                norm_feat = F.normalize(features, dim=-1).cpu().squeeze(0)
                
                # D. 存入字典 {文件名: 特征向量}
                filename = os.path.basename(img_path)
                vector_db[filename] = norm_feat
                success_count += 1
                
            except Exception as e:
                print(f"\n⚠️ Skipping {os.path.basename(img_path)}: {e}")

    # 5. 保存结果
    if success_count > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 'fake_image_vectors.pt')
        
        torch.save(vector_db, save_path)
        print(f"\n🎉 Database generated successfully!")
        print(f"💾 Saved to: {os.path.abspath(save_path)}")
        print(f"Dg Total Vectors: {len(vector_db)}")
        print("💡 提示: 现在你可以运行 explainable_api.py 来使用这个数据库了。")
    else:
        print("\n❌ 没有任何特征被提取，数据库未生成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Vector DB for Explainable Fake Detection (Test Version)")
    
    # 必填：你的图片文件夹路径
    parser.add_argument('--data_path', type=str, required=True, help='Path to your image folder (e.g., D:/Datasets/Test/Fake)')
    
    # 选填：保存路径 (默认当前目录)
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for .pt file')
    
    args = parser.parse_args()
    build_database(args)