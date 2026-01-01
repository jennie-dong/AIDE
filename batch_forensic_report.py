import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import base64
from torchvision import transforms, models
from openai import OpenAI
import open_clip
from tqdm import tqdm
import time

# =================配置区域=================
# 这样无论别人把文件夹放在哪里，BASE_DIR 都会自动变成那个路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔴 必填：上传 Github 前，请务必删除真实的 Key，换成提示语或环境变量！
API_KEY = "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"  # 🔴 必填：换成你的 Key

# 模型权重：os.path.join 会自动拼接路径，兼容 Windows 和 Mac
CKPT_PATH = os.path.join(BASE_DIR, "Final_Augmented_Food_Detector.pth")

# 向量数据库
DB_PATH = os.path.join(BASE_DIR, "fake_image_vectors.pt")

# 📂 输入：原始 AI 假图文件夹
RAW_IMG_DIR = os.path.join(BASE_DIR, "AI_JPG")

# 📂 输入：组长生成的特征图文件夹
PATCH_DIR = os.path.join(BASE_DIR, "eval_cam_results_db_bowen")

# 💾 输出：报告保存路径
OUTPUT_FILE = os.path.join(BASE_DIR, "FINAL_FORENSIC_REPORTS.txt")

# ==========================================
# --- 1. 定义检测模型 ---
class SRMConv2d(torch.nn.Module):
    def __init__(self, inc=3, outc=30):
        super(SRMConv2d, self).__init__()
        self.hpf = torch.nn.Conv2d(inc, outc, kernel_size=5, padding=2, bias=False)
    def forward(self, x): return self.hpf(x)

class SRMResNet(torch.nn.Module):
    def __init__(self):
        super(SRMResNet, self).__init__()
        self.hpf = SRMConv2d(3, 30)
        # weights=None 解决你之前的 warning
        self.model_min = models.resnet50(weights=None)
        self.model_min.conv1 = torch.nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model_min.fc = torch.nn.Linear(2048, 2) 
    def forward(self, x):
        x = self.hpf(x)
        x = self.model_min(x)
        return x

# --- 2. 定义语义提取器 ---
class SemanticExtractor:
    def __init__(self, device):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.to(device)
        self.model.eval()
        self.device = device
    def extract(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = F.normalize(features, dim=-1)
            return features
        except:
            return None

# --- 3. 定义 GPT 大脑 ---
class ForensicBrain:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    def encode_image(self, image_path):
        if not image_path or not os.path.exists(image_path): return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_case(self, case_data):
        base64_origin = self.encode_image(case_data['origin_path'])
        base64_patch = self.encode_image(case_data['patch_path'])
        
        import json
        
        # 1. 定义系统角色 (JSON结构)
        prompt_structure = {
            "role": "Senior Digital Forensic Expert (AIDE Framework)",
            "task": "Analyze visual evidence to determine authenticity. Do NOT rely on metadata.",
            "visual_evidence_legend": {
                "Red/Blue_Dots": "MAX Regions (High-Freq Artifacts): jagged edges, noise.",
                "Green/Yellow_Dots": "MIN Regions (Low-Freq Artifacts): waxy smoothness, texture loss."
            },
            "workflow": [
                "Step 1: Check Detector Confidence (Hard Evidence).",
                "Step 2: Analyze Feature Map (Microscopic Visual Artifacts).",
                "Step 3: Check Semantic Consistency (Database Matches).",
                "Step 4: Conclusion based on Physics & Logic."
            ]
        }
        system_content = json.dumps(prompt_structure, indent=2)

        # 2. 定义用户输入 (注意：这里不再包含 filename)
        # 变量名统一为 user_content_str，解决 NameError
        user_content_str = f"""
        [EVIDENCE DATA]
        - Detector Verdict: {case_data['verdict']} (Confidence: {case_data['conf']:.2%})
        - Database Retrieval: {case_data['semantic_info']}
        
        [INSTRUCTION]
        Analyze the Original Image and the Feature Map (if available).
        Focus on:
        1. Physical inconsistencies (Light, Shadow, Gravity).
        2. Frequency artifacts marked by the dots.
        3. Do NOT make assumptions based on the filename (it is hidden).
        """
        
        # 3. 构建消息列表
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "text", "text": user_content_str}, # 👈 这里现在能找到变量了
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_origin}"}},
            ]}
        ]
        
        if base64_patch:
            messages[1]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_patch}"}}
            )
            messages[1]["content"].append({"type": "text", "text": "(Image 2: AIDE Feature Map)"})
        else:
            messages[1]["content"].append({"type": "text", "text": "(Image 2: Feature Map Missing, analyze original only)"})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"GPT Error: {e}"
        
# --- 4. 主流程 ---
def main():
    device = "cpu" # 推荐 CPU 比较稳
    print(f"🚀 Starting Batch Forensic Analysis on {device}...")

    # A. 加载所有组件
    detector = SRMResNet()
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    detector.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    detector.to(device).eval()
    print("✅ Detector Loaded.")
    
    semantic_extractor = SemanticExtractor(device)
    print("✅ Semantic Extractor Loaded.")
    
    brain = ForensicBrain(API_KEY)
    
    vector_db = None
    if os.path.exists(DB_PATH):
        vector_db = torch.load(DB_PATH, map_location=device)
        print("✅ Vector DB Loaded.")
    
    # B. 准备数据转换
    trans = transforms.Compose([
        transforms.Resize([256, 256]), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # C. 扫描所有待测图片
    image_files = glob.glob(os.path.join(RAW_IMG_DIR, "*.*"))
    # 过滤非图片文件
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📂 Found {len(image_files)} images to process.")
    if len(image_files) == 0:
        print(f"❌ Error: No images in {RAW_IMG_DIR}")
        return

    # 清空或新建输出文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== AIDE FORENSIC BATCH REPORT ===\n\n")

    # D. 开始循环处理
    for img_path in tqdm(image_files, desc="Processing"):
        filename = os.path.basename(img_path)
        
        # 1. 检测 (Hard Evidence)
        try:
            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = trans(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = detector(input_tensor)
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred_idx].item()
                verdict = "FAKE" if pred_idx == 0 else "REAL"
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # 2. 匹配组长的特征图 (Visual Evidence)
        # 规则：去 bowen 文件夹找 patch_all_{name}.*
        name_no_ext = os.path.splitext(filename)[0]
        patch_path = None
        
        # 尝试几种可能的命名 (jpg, png, 有后缀, 无后缀)
        candidates = [
            f"patch_all_{filename}",        # patch_all_name.jpg
            f"patch_all_{filename}.png",    # patch_all_name.jpg.png
            f"patch_all_{name_no_ext}.png", # patch_all_name.png
            f"patch_all_{name_no_ext}.jpg"  # patch_all_name.jpg
        ]
        
        for cand in candidates:
            full_path = os.path.join(PATCH_DIR, cand)
            if os.path.exists(full_path):
                patch_path = full_path
                break
        
        # 3. 语义检索 (Semantic Evidence) - 已做防泄露处理
        semantic_info = "No database matches found."
        if vector_db:
            feat = semantic_extractor.extract(img_path)
            if feat is not None:
                sims = []
                feat = feat.detach().cpu()
                for db_name, db_feat in vector_db.items():
                    db_feat = db_feat.cpu()
                    if db_feat.shape[-1] == feat.shape[-1]:
                        sim = F.cosine_similarity(feat, db_feat).item()
                        sims.append((db_name, sim))
                
                # 取前3名
                top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
                
                # 🔴 关键修改：不再发送具体文件名，只发送相似度分数
                # 防止 GPT 通过数据库里的文件名（如 "spoiled_meat_fake.jpg"）直接猜到答案
                if top3:
                    top_score = top3[0][1]
                    if top_score > 0.8:
                        semantic_info = f"HIGH RISK: Found {len(top3)} similar cases in fake database. Top similarity score: {top_score:.2f} (Matches known fraud patterns)."
                    elif top_score > 0.6:
                        semantic_info = f"MEDIUM RISK: Moderate similarity to known fakes (Score: {top_score:.2f})."
                    else:
                        semantic_info = f"LOW RISK: Low similarity to database entries (Score: {top_score:.2f})."

        # 4. GPT 生成报告
        report = brain.analyze_case({
            'filename': filename,
            'origin_path': img_path,
            'patch_path': patch_path,
            'verdict': verdict,
            'conf': conf,
            'semantic_info': semantic_info
        })

        # 5. 写入文件
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Case: {filename} ---\n")
            f.write(f"Patch Map Found: {'YES' if patch_path else 'NO'}\n")
            f.write(report + "\n\n")
            f.write("="*50 + "\n\n")
        
        # 休息一下防止 API 限流
        time.sleep(1)

    print(f"\n🎉 All done! Reports saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()