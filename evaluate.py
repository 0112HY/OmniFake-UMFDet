
# import logging
# import random
# import inspect
# import torch
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm

# from transformers import AutoModelForCausalLM, AutoProcessor
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     classification_report,
#     confusion_matrix,
# )

# from dataset_backbone import FakeDataset,PROMPT_TEMPLATE

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model_Path = "/data1/hy/UMFDet_ckp/backbone_ckp/florence_backbone_only_qkvfclm/epoch_4"
# model = AutoModelForCausalLM.from_pretrained(
#     Model_Path,
#     trust_remote_code=True
# ).to(device)

# processor = AutoProcessor.from_pretrained(
#     Model_Path,
#     trust_remote_code=True
# )

# def run_example(task_prompt, text_input, image):
#     prompt = task_prompt + text_input
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         max_new_tokens=1024,
#         num_beams=3,
#     )
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
#     parsed_answer = processor.post_process_generation(
#         generated_text, task=task_prompt, image_size=(image.width, image.height)
#     )
#     return parsed_answer

# def collate_fn(batch):
#     questions, answers, images = zip(*batch)
#     inputs = processor(
#         text=list(questions), images=list(images), return_tensors="pt", padding=True
#     ).to(device)
#     return inputs, answers

# # ---------------------- Dataset & Dataloader ----------------------
# batch_size = 4
# num_workers = 0
# prefetch_factor = None

# test_dataset = FakeDataset(
#     csv_file_path="/data1/hy/OmniFake_final_10K_with_visual/test_final.csv",
#     image_directory_path="/data1/hy/OmniFake_final_10K_with_visual/"
# )

# # 如果只想抽样测试，放开下面三行
# # subset_size = int(0.2 * len(test_dataset))
# # indices = random.sample(range(len(test_dataset)), subset_size)
# # test_dataset = Subset(test_dataset, indices)

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=batch_size,
#     collate_fn=collate_fn,
#     num_workers=num_workers,
#     prefetch_factor=prefetch_factor,
# )

# # ---------------------- Inference helpers ----------------------
# def run_batch(inputs):
#     generated_ids = model.generate(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         max_new_tokens=1024,
#         num_beams=3,
#     )
#     generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
#     return generated_texts

# # ---------------------- Evaluation ----------------------
# # ---------------------- Full Evaluation Preview (No Metrics) ----------------------
# def evaluate_model_preview(test_loader):
#     """
#     遍历完整的验证集，只做推理预览：查看所有 batch 的模型原始输出和解析结果
#     """
#     task_prompt = "<DETECTION_NEWS>"
    
#     print("\n🚀 开始全量模型输出格式预览 (仅预览，不计算指标)...\n")

#     for batch_idx, (inputs, batch_answers) in enumerate(tqdm(test_loader, desc="Generating")):
#         generated_texts = run_batch(inputs)
#         H, W = inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]

#         # 如果输出太多，你可以选择只打印每个 batch 的第一个，或者全部打印
#         print(f"\n========== Batch {batch_idx + 1} ==========")
#         for i, (raw_text, gt_answer) in enumerate(zip(generated_texts, batch_answers)):
            
#             # 1. Florence-2 post_process_generation 解析
#             parsed_answer = processor.post_process_generation(
#                 raw_text,
#                 task=task_prompt,
#                 image_size=(W, H),
#             )
            
#             # 安全获取解析内容（如果 task_prompt 没对上，给个默认回退）
#             pred = parsed_answer.get(task_prompt, raw_text).replace("<pad>", "").replace("</s>", "").strip()
#             gt = gt_answer.strip()

#             print(f"--- Sample {i + 1} ---")
#             print(f"[Ground Truth] : {gt}")
#             print(f"[Raw Output]   : {raw_text}")  
#             print(f"[Parsed Pred]  : {pred}")    

#     print("\n✅ 全量预览结束。")

# # ---------------------- Run ----------------------
# if __name__ == "__main__":
#     # 调用全量预览函数，不计算指标
#     evaluate_model_preview(test_loader)

# # ---------------------- Run ----------------------
# # if __name__ == "__main__":
# #     acc = evaluate_model(test_loader)
# #     print(f"\nFINAL ALL Accuracy: {acc:.4f}")




import logging
import random
import inspect
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset_backbone import FakeDataset, PROMPT_TEMPLATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model_Path = "/data1/hy/UMFDet_ckp/backbone_ckp/backbone_frozenVit_5class_moe_0.05edr_0.01cca/epoch_9"
model = AutoModelForCausalLM.from_pretrained(
    Model_Path,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained(
    Model_Path,
    trust_remote_code=True
)

def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    ).to(device)
    return inputs, answers

# ---------------------- Dataset & Dataloader ----------------------
batch_size = 4
num_workers = 0
prefetch_factor = None

test_dataset = FakeDataset(
    csv_file_path="/data1/hy/OmniFake_final_10K_with_visual/test_final_balanced_clean_with_bbox.csv",
    image_directory_path="/data1/hy/OmniFake_final_10K_with_visual/"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
)

# ---------------------- Inference helpers ----------------------
def run_batch(inputs):
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return generated_texts

# ---------------------- Evaluation (5分类 & 2分类) ----------------------
def evaluate_model(test_loader):
    """
    遍历完整的验证集，实时打印推理结果，并最终计算5分类和2分类的各项指标。
    """
    task_prompt = "<DETECTION_NEWS>"
    
    y_true_5class = []
    y_pred_5class = []
    
    y_true_2class = []
    y_pred_2class = []
    
    print("\n🚀 开始评估模型并计算指标...\n")

    for batch_idx, (inputs, batch_answers) in enumerate(tqdm(test_loader, desc="Evaluating")):
        generated_texts = run_batch(inputs)
        H, W = inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]

        print(f"\n========== Batch {batch_idx + 1} ==========")
        for i, (raw_text, gt_answer) in enumerate(zip(generated_texts, batch_answers)):
            # 1. Florence-2 解析
            parsed_answer = processor.post_process_generation(
                raw_text,
                task=task_prompt,
                image_size=(W, H),
            )
            
            # 2. 提取预测结果并清理字符串格式
            if isinstance(parsed_answer, dict):
                pred = parsed_answer.get(task_prompt, raw_text)
            else:
                pred = raw_text
                
            # 获取原始的预测结果和GT用于打印展示
            pred_print = pred.replace("<pad>", "").replace("</s>", "").strip()
            gt_print = gt_answer.strip()
            
            # 实时打印中间结果
            print(f"--- Sample {i + 1} ---")
            print(f"[Ground Truth] : {gt_print}")
            print(f"[Raw Output]   : {raw_text}")  
            print(f"[Parsed Pred]  : {pred_print}")    

            # 转小写用于严格匹配计算指标
            pred_clean = pred_print.lower()
            gt_clean = gt_print.lower()

            # ------------------ 5 分类收集 ------------------
            y_true_5class.append(gt_clean)
            y_pred_5class.append(pred_clean)
            
            # ------------------ 2 分类收集 (Real vs Fake) ------------------
            # 逻辑：只要 Ground Truth 里包含 'real'，就是真新闻，其他四类全算 'fake'
            gt_2class = "real" if "real" in gt_clean else "fake"
            
            # 预测同理：如果模型预测包含 'real'，就是预测为真，否则预测为假
            pred_2class = "real" if "real" in pred_clean else "fake"
            
            y_true_2class.append(gt_2class)
            y_pred_2class.append(pred_2class)

    # ========================== 指标计算与打印 ==========================
    
    print("\n" + "="*60)
    print(" 📊 五分类评估报告 (5-Class Evaluation)")
    print("="*60)
    acc_5 = accuracy_score(y_true_5class, y_pred_5class)
    print(f"✅ Overall 5-Class Accuracy: {acc_5:.4f}\n")
    print("🔍 细粒度指标 (Per-class Precision, Recall, F1):")
    # classification_report 会自动列出每一类的 Precision, Recall, F1 和样本数(Support)
    print(classification_report(y_true_5class, y_pred_5class, digits=4))
    
    print("\n" + "="*60)
    print(" 📊 二分类评估报告 (2-Class: Real vs Fake)")
    print("="*60)
    acc_2 = accuracy_score(y_true_2class, y_pred_2class)
    print(f"✅ Overall 2-Class Accuracy: {acc_2:.4f}\n")
    print("🔍 细粒度指标 (Per-class Precision, Recall, F1):")
    print(classification_report(y_true_2class, y_pred_2class, digits=4))
    
    # 打印 5分类的混淆矩阵
    print("\n" + "="*60)
    print(" 🧠 五分类混淆矩阵 (Confusion Matrix)")
    print("="*60)
    labels_5 = sorted(list(set(y_true_5class))) # 获取所有出现的标签
    cm = confusion_matrix(y_true_5class, y_pred_5class, labels=labels_5)
    print(f"标签顺序: {labels_5}")
    print(cm)

    return acc_5, acc_2

# ---------------------- Run ----------------------
if __name__ == "__main__":
    # 调用评估函数并计算指标
    evaluate_model(test_loader)