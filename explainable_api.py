# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# import numpy as np
# import cv2
# import base64
# from torchvision import transforms, models
# from openai import OpenAI

# # 🔥 关键新增：引入 open_clip 库来弥补缺失的语义分支
# import open_clip 

# # ==========================================
# # 1. 左手：你的外卖检测专家 (SRMResNet)
# #    (负责：判定真假 + 生成热力图)
# # ==========================================
# class SRMConv2d(nn.Module):
#     def __init__(self, inc=3, outc=30):
#         super(SRMConv2d, self).__init__()
#         self.hpf = nn.Conv2d(inc, outc, kernel_size=5, padding=2, bias=False)
#     def forward(self, x): return self.hpf(x)

# class SRMResNet(nn.Module):
#     def __init__(self):
#         super(SRMResNet, self).__init__()
#         self.hpf = SRMConv2d(3, 30)
#         self.model_min = models.resnet50(weights=None)
#         self.model_min.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model_min.fc = nn.Linear(2048, 2) 
#     def forward(self, x):
#         x = self.hpf(x)
#         x = self.model_min(x)
#         return x

# # ==========================================
# # 2. 右手：语义特征提取器 (OpenCLIP)
# #    (负责：提取特征 + 数据库检索)
# #    我们直接加载官方预训练模型，不需要你自己练！
# # ==========================================
# class SemanticExtractor:
#     def __init__(self, device):
#         print("🌐 Initializing Semantic Extractor (OpenCLIP)...")
#         # 使用 ViT-B-32，它是最常用的 CLIP 模型，速度快，效果好
#         # 第一次运行时会自动下载权重 (约300MB)
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#         self.model.to(device)
#         self.model.eval()
#         self.device = device

#     def extract(self, image_path):
#         """读取图片并提取 512 维语义向量"""
#         image = Image.open(image_path).convert('RGB')
#         # 使用 CLIP 自己的预处理
#         image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             features = self.model.encode_image(image_tensor)
#             features = F.normalize(features, dim=-1) # 归一化
#         return features

# # ==========================================
# # 3. 大脑：GPT-4o 法医分析师
# #    (负责：看图 + 看热力图 + 看检索结果 -> 写报告)
# # ==========================================
# class ForensicBrain:
#     def __init__(self, api_key):
#         self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

#     def encode_image(self, image_path):
#         if not os.path.exists(image_path): return None
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')

#     def generate_report(self, case_data):
#         base64_origin = self.encode_image(case_data['original_img_path'])
#         base64_cam = self.encode_image(case_data['cam_img_path'])
        
#         system_prompt = """
#         You are an AI Forensic Expert. Combine "Pixel Artifacts" and "Semantic Logic" to verify images.
        
#         Your Analysis Process:
#         1. **Artifacts (Hard Evidence):** Check the Heatmap. Red areas = algorithm traces detected by SRM-ResNet.
#         2. **Semantics (Knowledge):** The system has retrieved similar known AI cases from the database based on semantic features.
#         3. **Visual Logic:** Check the Original Image for physical/lighting flaws.
        
#         Output a professional report explaining WHY it is Fake.
#         """
        
#         user_prompt = f"""
#         [Case Data]
#         - Detection Verdict: {case_data['aide_verdict']} (Confidence: {case_data['aide_conf']:.2%})
#         - Semantic Retrieval: The image is semantically similar to these known generation patterns:
#           {case_data['similar_evidence']}
        
#         Please analyze the Original Image and the Artifact Heatmap.
#         """
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": [
#                 {"type": "text", "text": user_prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_origin}"}},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_cam}"} if base64_cam else {"type": "text", "text": "No heatmap."}}
#             ]}
#         ]
        
#         print("🤖 Calling GPT-4o...")
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o", messages=messages, temperature=0.3
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"GPT Error: {e}"

# # ==========================================
# # 4. 总指挥：可解释性检测系统
# # ==========================================
# class ExplainableDetector:
#     def __init__(self, model_path, db_path, api_key, patch_dir, device='cpu'):
#         self.device = torch.device(device)
#         print(f"🚀 Initializing System on {self.device}...")
        
#         # 1. 你的检测模型 (保持不变)
#         self.detector = SRMResNet()
#         self._load_detector_weights(model_path)
#         self.detector.to(self.device)
#         self.detector.eval()
        
#         # 2. 语义提取 (保持不变)
#         self.semantic_extractor = SemanticExtractor(self.device)
        
#         # 3. GPT 大脑 (保持不变)
#         self.brain = ForensicBrain(api_key)
        
#         # 4. 数据库 (保持不变)
#         self.vector_db = None
#         if os.path.exists(db_path):
#             self.vector_db = torch.load(db_path, map_location=self.device)
            
#         # 5. [新增] 组长的特征图文件夹
#         self.patch_dir = patch_dir
#         if not os.path.exists(self.patch_dir):
#             print(f"⚠️ Warning: Patch directory not found: {self.patch_dir}")

#         self.trans = transforms.Compose([
#             transforms.Resize([256, 256]),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def _load_detector_weights(self, path):
#         # ... (保持不变) ...
#         if not os.path.exists(path): raise FileNotFoundError(f"Model not found: {path}")
#         ckpt = torch.load(path, map_location='cpu')
#         state = ckpt['model'] if 'model' in ckpt else ckpt
#         new_state = {k.replace("module.", ""): v for k, v in state.items()}
#         self.detector.load_state_dict(new_state, strict=False) 
#         print("✅ Food Detector weights loaded.")

#     def search_db(self, query_feat):
#         # ... (保持不变) ...
#         if not self.vector_db: return "Database not available."
#         query_feat = query_feat.detach().cpu()
#         sims = []
#         for name, db_feat in self.vector_db.items():
#             db_feat = db_feat.cpu()
#             if db_feat.shape[-1] != query_feat.shape[-1]: continue 
#             sim = F.cosine_similarity(query_feat, db_feat).item()
#             sims.append((name, sim))
#         top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
#         return ", ".join([f"{n} ({s:.2f})" for n, s in top3])

#     def get_patch_image_path(self, original_image_path):
#         """
#         根据原图路径，去组长的文件夹里找对应的 patch_all 图片
#         假设规则：原图 xxx.jpg -> 组长图 patch_all_xxx.jpg
#         """
#         filename = os.path.basename(original_image_path)
#         # ⚠️ 请根据组长的实际命名规则修改这里
#         patch_filename = f"patch_all_{filename}" 
#         # 或者如果组长只是保持原名： patch_filename = filename
        
#         full_path = os.path.join(self.patch_dir, patch_filename)
#         if os.path.exists(full_path):
#             return full_path
#         else:
#             print(f"⚠️ 没找到对应的特征图: {full_path}")
#             return None

#     def run(self, image_path):
#         print(f"\n🔍 Processing: {image_path}")
        
#         # 1. 视觉检测 (Run Your Model)
#         img_pil = Image.open(image_path).convert('RGB')
#         input_tensor = self.trans(img_pil).unsqueeze(0).to(self.device)
        
#         self.detector.zero_grad()
#         logits = self.detector(input_tensor)
#         probs = F.softmax(logits, dim=1)
#         pred_idx = torch.argmax(probs, dim=1).item()
#         conf = probs[0][pred_idx].item()
        
#         # 假设 0=Fake
#         verdict = "FAKE" if pred_idx == 0 else "REAL"
#         print(f"   👉 Verdict: {verdict} ({conf:.2%})")

#         # 2. 获取证据图 (使用组长的图！)
#         evidence_path = self.get_patch_image_path(image_path)
        
#         # 如果找不到组长的图，就只用原图，或者你可以保留之前的 Heatmap 逻辑作为备选
#         if evidence_path:
#             print(f"   📸 Found AIDE Feature Map: {os.path.basename(evidence_path)}")
#         else:
#             print("   ⚠️ Using Original Image only (Feature map missing).")
#             evidence_path = None

#         # 3. 语义检索 (保持不变)
#         semantic_feat = self.semantic_extractor.extract(image_path)
#         sim_result = self.search_db(semantic_feat)

#         # 4. GPT-4o 生成报告 (Prompt 需要微调以适应新图)
#         # 我们需要在 prompt 里解释 Max/Min 的含义
#         self.brain_generate_report_v2(image_path, evidence_path, verdict, conf, sim_result)

#     def brain_generate_report_v2(self, original_path, evidence_path, verdict, conf, sim_result):
#         # 专门针对 AIDE 特征图的 Prompt
#         system_prompt = """
#         You are an AI Forensic Expert. You analyze images using "Frequency Domain Artifacts" (AIDE method).
        
#         Input Images:
#         1. Original Image.
#         2. **Artifact Feature Map**: This image highlights specific patches based on frequency analysis:
#            - **Max Regions (High Frequency):** Look for unnatural sharp edges, noise patterns, or "jagged" artifacts.
#            - **Min Regions (Low Frequency):** Look for unnatural smoothness, blurring, or texture loss.
        
#         Task:
#         - Analyze the "Feature Map" to identify where the algorithm detected anomalies.
#         - Combine this with the "Detection Verdict" and "Semantic Retrieval" results.
#         - Explain WHY the image is Real or Fake based on these specific regions.
#         """
        
#         user_prompt = f"""
#         [Case Data]
#         - Verdict: {verdict} (Confidence: {conf:.2%})
#         - Semantic Retrieval: Similar to {sim_result}
        
#         Please analyze the Original Image and the provided Feature Map. Focus on the Max/Min artifact regions.
#         """
        
#         # 调用 GPT (复用之前的逻辑，只是换了 prompt)
#         # 这里为了简洁直接把 brain 的逻辑搬过来或者调用 brain 的方法
#         # 假设你还在用之前的 ForensicBrain 类，这里稍作适配
#         base64_origin = self.brain.encode_image(original_path)
#         base64_evidence = self.brain.encode_image(evidence_path) if evidence_path else None
        
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": [
#                 {"type": "text", "text": user_prompt},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_origin}"}},
#             ]}
#         ]
        
#         if base64_evidence:
#             messages[0]["content"] += "\n(The second image is the Artifact Feature Map)"
#             messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_evidence}"}})
            
#         print("🤖 Calling GPT-4o...")
#         try:
#             response = self.brain.client.chat.completions.create(
#                 model="gpt-4o", messages=messages, temperature=0.3
#             )
#             print("\n📝 FINAL REPORT:\n" + response.choices[0].message.content)
#         except Exception as e:
#             print(f"GPT Error: {e}")
            
# if __name__ == "__main__":
#     # 配置
#     API_KEY = "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" # 替换你的 Key
#     CKPT = r"D:\北大\深度学习\大作业\Final_Augmented_Food_Detector.pth"
#     DB = "fake_image_vectors.pt"
    
#     # 图片路径
#     IMG = r"D:\北大\深度学习\大作业\AI外卖测试\jimeng-2025-12-19-9443-过期的罐头食品，罐头表面有锈迹，标签发黄褪色，有些罐头盖子鼓起变形，自然写实摄影....jpg"
    
#     if os.path.exists(IMG):
#         # 只要你有 GPU 驱动，这里可以用 'cuda'，否则 'cpu'
#         detector = ExplainableDetector(CKPT, DB, API_KEY, device='cpu')
#         detector.run(IMG)
#     else:
#         print("Image not found!")
import os
import shutil
import subprocess
import torch
import torch.nn.functional as F
from PIL import Image
import base64
from torchvision import transforms, models
from openai import OpenAI
import open_clip
import glob
import time

# ==========================================
# 1. 辅助函数：自动化运行组长的 Eval 流程
# ==========================================
def run_leader_eval_process(original_image_path, project_root, model_ckpt):
    """
    全自动流水线：
    1. 建文件夹 -> 2. 移图片 -> 3. 跑命令 -> 4. 拿结果
    """
    print(f"🏭 [Pipeline] Starting AIDE Feature Generation for: {os.path.basename(original_image_path)}")
    
    # --- A. 准备所有的路径 ---
    # 组长要求的结构: Data/eval/test_set/1_fake/图片
    eval_dir = os.path.join(project_root, "Data", "eval", "test_set", "1_fake")
    train_dir = os.path.join(project_root, "Data", "train", "dummy", "1_fake") # 骗过bug用的
    output_dir = os.path.join(project_root, "bowen_results") # 结果存这里
    
    # 清理并重建临时目录 (保证每次只跑这一张图)
    if os.path.exists(os.path.join(project_root, "Data")):
        shutil.rmtree(os.path.join(project_root, "Data")) # 暴力清理旧数据
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- B. 把图片复制进去 ---
    target_path = os.path.join(eval_dir, os.path.basename(original_image_path))
    shutil.copy(original_image_path, target_path)
    
    # --- C. 拼凑运行命令 ---
    # 这就是把组长的 eval.sh 翻译成 Python 调用
    # 注意：我们必须切换工作目录到项目根目录，否则找不到 main_finetune.py
    cmd = [
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node", "1",
        "--master_port", "29505", # 换个端口防止冲突
        "main_finetune.py",
        "--model", "AIDE",
        "--batch_size", "1",
        "--blr", "5e-4",
        "--epochs", "1", # 跑1轮就够了
        "--data_path", "Data/train",      # 传个空路径骗过它
        "--eval_data_path", "Data/eval",  # 真实的图片在这里
        "--resume", model_ckpt,           # 你的权重
        "--eval", "True",                 # 开启评测模式
        "--output_dir", "bowen_results"   # 输出路径
    ]
    
    print("   ⚙️ Running AIDE Eval Command (This may take a few seconds)...")
    try:
        # 执行命令 (cwd=project_root 确保在项目根目录下运行)
        subprocess.run(cmd, cwd=project_root, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Eval command failed: {e}")
        return None

    # --- D. 找生成的图 ---
    # 组长生成的图通常在 output_dir 下面，或者是 output_dir/patch_all_xxx.png
    # 我们遍历找一下
    patch_name = f"patch_all_{os.path.basename(original_image_path)}"
    # 有时候它是 png，有时候保留原后缀，我们要灵活一点
    search_pattern = os.path.join(output_dir, "patch_all_*")
    found_files = glob.glob(search_pattern)
    
    # 简单的匹配逻辑：找最新的那张，或者名字匹配的
    if found_files:
        # 假设就是最新生成的那张
        latest_file = max(found_files, key=os.path.getctime)
        print(f"   📸 Feature Map Generated: {latest_file}")
        return latest_file
    
    print("   ⚠️ No output image found.")
    return None

# ==========================================
# 2. 你的检测模型 (SRMResNet) - 负责打分
# ==========================================
class SRMConv2d(torch.nn.Module):
    def __init__(self, inc=3, outc=30):
        super(SRMConv2d, self).__init__()
        self.hpf = torch.nn.Conv2d(inc, outc, kernel_size=5, padding=2, bias=False)
    def forward(self, x): return self.hpf(x)

class SRMResNet(torch.nn.Module):
    def __init__(self):
        super(SRMResNet, self).__init__()
        self.hpf = SRMConv2d(3, 30)
        self.model_min = models.resnet50(weights=None)
        self.model_min.conv1 = torch.nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model_min.fc = torch.nn.Linear(2048, 2) 
    def forward(self, x):
        x = self.hpf(x)
        x = self.model_min(x)
        return x

# ==========================================
# 3. 语义特征提取 (OpenCLIP) - 负责找同伙
# ==========================================
class SemanticExtractor:
    def __init__(self, device):
        print("🌐 Initializing Semantic Extractor (OpenCLIP)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.to(device)
        self.model.eval()
        self.device = device

    def extract(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = F.normalize(features, dim=-1)
        return features

# ==========================================
# 4. GPT-4o 大脑 - 负责写报告
# ==========================================
class ForensicBrain:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    def encode_image(self, image_path):
        if not image_path or not os.path.exists(image_path): return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_report(self, case_data):
        base64_origin = self.encode_image(case_data['original_img_path'])
        base64_evidence = self.encode_image(case_data['evidence_path'])
        
        system_prompt = """
        You are an AI Forensic Expert using the AIDE framework.
        Interpreting the AIDE Feature Map (Image 2):
        - **Red/Blue Dots (Max Regions):** High-Frequency anomalies (sharp edges, noise).
        - **Green/Yellow Dots (Min Regions):** Low-Frequency anomalies (unnatural smoothness).
        
        Task: Write a forensic report. combine the Detector Verdict, Semantic Retrieval, and visual analysis of the dots.
        """
        
        user_prompt = f"""
        [Case Data]
        - Verdict: {case_data['aide_verdict']} ({case_data['aide_conf']:.2%})
        - Semantic Retrieval: {case_data['similar_evidence']}
        
        Analyze the Feature Map (Image 2). Explain why the model marked those specific spots.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_origin}"}},
            ]}
        ]
        
        if base64_evidence:
            messages[1]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_evidence}"}}
            )
            messages[1]["content"].append({"type": "text", "text": "(This is the AIDE Feature Map)"})
        else:
            messages[1]["content"].append({"type": "text", "text": "(Feature Map missing, analyze original only)"})

        print("🤖 Calling GPT-4o...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"GPT Error: {e}"

# ==========================================
# 5. 总指挥
# ==========================================
class ExplainableDetector:
    def __init__(self, model_path, db_path, api_key, project_root, device='cpu'):
        self.device = torch.device(device)
        self.project_root = project_root
        self.model_path = model_path # 保存下来给 subprocess 用
        print(f"🚀 Initializing System on {self.device}...")
        
        self.detector = SRMResNet()
        self._load_detector_weights(model_path)
        self.detector.to(self.device)
        self.detector.eval()
        
        self.semantic_extractor = SemanticExtractor(self.device)
        self.brain = ForensicBrain(api_key)
        self.vector_db = torch.load(db_path, map_location=self.device) if os.path.exists(db_path) else None
        
        self.trans = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_detector_weights(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Model not found: {path}")
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        self.detector.load_state_dict(new_state, strict=False) 
        print("✅ Food Detector weights loaded.")

    def search_db(self, query_feat):
        if not self.vector_db: return "Database not available."
        query_feat = query_feat.detach().cpu()
        sims = []
        for name, db_feat in self.vector_db.items():
            db_feat = db_feat.cpu()
            if db_feat.shape[-1] != query_feat.shape[-1]: continue 
            sim = F.cosine_similarity(query_feat, db_feat).item()
            sims.append((name, sim))
        top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        return ", ".join([f"{n} ({s:.2f})" for n, s in top3])

    def run(self, image_path):
        print(f"\n🔍 Processing: {image_path}")
        
        # 1. 你的模型检测 (Hard Evidence)
        img_pil = Image.open(image_path).convert('RGB')
        input_tensor = self.trans(img_pil).unsqueeze(0).to(self.device)
        self.detector.zero_grad()
        logits = self.detector(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred_idx].item()
        verdict = "FAKE" if pred_idx == 0 else "REAL" 
        print(f"   👉 Verdict: {verdict} ({conf:.2%})")

        # 2. 调用组长的脚本生成特征图 (The "X-Ray")
        # 这一步是全自动的，你不需要手动移文件
        evidence_path = run_leader_eval_process(image_path, self.project_root, self.model_path)

        # 3. 语义检索 (Semantic Evidence)
        semantic_feat = self.semantic_extractor.extract(image_path)
        sim_result = self.search_db(semantic_feat)
        if sim_result: print(f"   📚 Retrieval: {sim_result[:50]}...")

        # 4. 生成报告
        report = self.brain.generate_report({
            'original_img_path': image_path,
            'evidence_path': evidence_path, 
            'aide_verdict': verdict,
            'aide_conf': conf,
            'similar_evidence': sim_result
        })
        
        print("\n📝 FINAL REPORT:\n" + report)

if __name__ == "__main__":
    # === 必填配置 ===
    API_KEY = "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"
    
    # 你的模型路径
    CKPT = r"D:\北大\深度学习\大作业\Final_Augmented_Food_Detector.pth"
    # 数据库路径
    DB = "fake_image_vectors.pt"
    # 项目根目录 (非常重要！就是含有 main_finetune.py 的那个文件夹)
    PROJECT_ROOT = r"D:\北大\深度学习\大作业\AIDE-main\AIDE-main"
    
    # 你要测试的图片 (随便放哪里都行，程序会自动把它复制到该去的地方)
    IMG = r"D:\北大\深度学习\大作业\AI外卖测试\jimeng-2025-12-19-8882-洒出的便当盒，塑料便当盒盖子没有盖紧，里面的饭菜洒出到盒子外面，米饭和配菜散落，....jpg"
    
    if os.path.exists(IMG):
        # 初始化 detector (注意 device='cpu' 比较稳妥)
        detector = ExplainableDetector(CKPT, DB, API_KEY, PROJECT_ROOT, device='cpu')
        detector.run(IMG)
    else:
        print("Image not found!")