
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
# ---------------------- Model & Processor ----------------------
Model_Path = "" 
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
#Test dataset path and image directory path should be specified here
test_dataset = FakeDataset(
    csv_file_path="",
    image_directory_path=""
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

# ---------------------- Evaluation (5 & 2) ----------------------
def evaluate_model(test_loader):

    task_prompt = "<DETECTION_NEWS>"
    
    y_true_5class = []
    y_pred_5class = []
    
    y_true_2class = []
    y_pred_2class = []
    
    print("\n🚀 evaluate being...\n")

    for batch_idx, (inputs, batch_answers) in enumerate(tqdm(test_loader, desc="Evaluating")):
        generated_texts = run_batch(inputs)
        H, W = inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]

        print(f"\n========== Batch {batch_idx + 1} ==========")
        for i, (raw_text, gt_answer) in enumerate(zip(generated_texts, batch_answers)):
            parsed_answer = processor.post_process_generation(
                raw_text,
                task=task_prompt,
                image_size=(W, H),
            )
            
            if isinstance(parsed_answer, dict):
                pred = parsed_answer.get(task_prompt, raw_text)
            else:
                pred = raw_text
                
            pred_print = pred.replace("<pad>", "").replace("</s>", "").strip()
            gt_print = gt_answer.strip()
            
            print(f"--- Sample {i + 1} ---")
            print(f"[Ground Truth] : {gt_print}")
            print(f"[Raw Output]   : {raw_text}")  
            print(f"[Parsed Pred]  : {pred_print}")    

            pred_clean = pred_print.lower()
            gt_clean = gt_print.lower()

            y_true_5class.append(gt_clean)
            y_pred_5class.append(pred_clean)
            
            gt_2class = "real" if "real" in gt_clean else "fake"
            
            pred_2class = "real" if "real" in pred_clean else "fake"
            
            y_true_2class.append(gt_2class)
            y_pred_2class.append(pred_2class)

    
    print("\n" + "="*60)
    print(" 📊 (5-Class Evaluation)")
    print("="*60)
    acc_5 = accuracy_score(y_true_5class, y_pred_5class)
    print(f"✅ Overall 5-Class Accuracy: {acc_5:.4f}\n")
    print("🔍  (Per-class Precision, Recall, F1):")
    print(classification_report(y_true_5class, y_pred_5class, digits=4))
    
    print("\n" + "="*60)
    print(" 📊  (2-Class: Real vs Fake)")
    print("="*60)
    acc_2 = accuracy_score(y_true_2class, y_pred_2class)
    print(f"✅ Overall 2-Class Accuracy: {acc_2:.4f}\n")
    print("🔍  (Per-class Precision, Recall, F1):")
    print(classification_report(y_true_2class, y_pred_2class, digits=4))
    
    print("\n" + "="*60)
    print(" 🧠  (Confusion Matrix)")
    print("="*60)
    labels_5 = sorted(list(set(y_true_5class))) 
    cm = confusion_matrix(y_true_5class, y_pred_5class, labels=labels_5)
    print(f"LABELS: {labels_5}")
    print(cm)

    return acc_5, acc_2

# ---------------------- Run ----------------------
if __name__ == "__main__":
    evaluate_model(test_loader)