import argparse
import os
import re
from functools import partial

import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
import random
from peft import LoraConfig, get_peft_model
from dataset_backbone import FakeDataset, PROMPT_TEMPLATE



TASK_PROMPT = "<DETECTION_NEWS>"
CLASS_NAME_TO_ID = {
    "real": 0,
    "rumor": 1,
    "vision_manipulation": 2,
    "text_manipulation": 3,
    "mixed_manipulation": 4,
}
CLASS_NAMES = tuple(CLASS_NAME_TO_ID.keys())
FAKE_CLASSES = {"rumor", "vision_manipulation", "text_manipulation", "mixed_manipulation"}
EDR_LOSS_WEIGHT = 0.05
CCA_LOSS_WEIGHT = 0.01


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    for token in ("<pad>", "</s>", "<s>", TASK_PROMPT.lower()):
        text = text.replace(token, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.:,;\"'[]{}()")


def extract_class_label(value):
    text = normalize_text(value)
    if not text:
        return None

    if "the news belongs to" in text:
        text = text.rsplit("the news belongs to", 1)[-1].strip(" \t\n\r.:,;")

    compact = re.sub(r"[\s-]+", "_", text.strip(" \t\n\r.:,;\"'[]{}()"))
    if compact in CLASS_NAME_TO_ID:
        return compact

    matches = []
    for label in CLASS_NAMES:
        variants = (label, label.replace("_", " "), label.replace("_", "-"))
        for variant in variants:
            pattern = r"(?<![a-z0-9_])" + re.escape(variant) + r"(?![a-z0-9_])"
            match = re.search(pattern, text)
            if match:
                matches.append((match.start(), -len(label), label))

    if not matches:
        return None

    matches.sort()
    return matches[0][2]


def class_to_binary_id(label):
    if label == "real":
        return 0
    if label in FAKE_CLASSES:
        return 1
    return None


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def answers_to_expert_labels(answers, device):
    label_ids = []
    for answer in answers:
        label = extract_class_label(answer)
        label_ids.append(CLASS_NAME_TO_ID.get(label, -100))
    return torch.tensor(label_ids, dtype=torch.long, device=device)


def parse_generated_label(raw_text, processor, image_size):
    parsed_text = raw_text
    try:
        parsed_answer = processor.post_process_generation(raw_text, task=TASK_PROMPT, image_size=image_size)
        if isinstance(parsed_answer, dict):
            parsed_text = parsed_answer.get(TASK_PROMPT, raw_text)
        elif parsed_answer is not None:
            parsed_text = parsed_answer
    except Exception:
        parsed_text = raw_text

    pred_label = extract_class_label(parsed_text)
    if pred_label is None:
        pred_label = extract_class_label(raw_text)
    return pred_label


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(device)
    return inputs, answers


def create_data_loaders(
    train_dataset,
    val_datasets,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        sampler=train_sampler,
    )

    val_loaders = {}
    for name, val_dataset in val_datasets.items():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size//2,
            collate_fn=partial(collate_fn, processor=processor, device=device),
            num_workers=num_workers,
            sampler=val_sampler,
        )
        val_loaders[name] = val_loader

    return train_loader, val_loaders


def evaluate_model(rank, world_size, model, val_loaders, device, train_loss, processor, global_step, batch_size, max_val_item_count):
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    final_metrics = None

    for val_name, val_loader in val_loaders.items():
        val_loss_sum = 0.0
        val_item_count = 0
        correct_5 = 0
        correct_2 = 0
        invalid_pred_count = 0
        tp_5 = [0] * len(CLASS_NAMES)
        fp_5 = [0] * len(CLASS_NAMES)
        fn_5 = [0] * len(CLASS_NAMES)
        tp_2 = [0] * 2
        fp_2 = [0] * 2
        fn_2 = [0] * 2

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                inputs, answers = batch

                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)

                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(device)

                outputs = eval_model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                batch_item_count = len(answers)
                val_loss_sum += outputs.loss.item() * batch_item_count
                val_item_count += batch_item_count

                generated_ids = eval_model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=32,
                    num_beams=3,
                    do_sample=False,
                )
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                image_size = (pixel_values.shape[-1], pixel_values.shape[-2])

                for raw_text, answer in zip(generated_texts, answers):
                    true_label = extract_class_label(answer)
                    pred_label = parse_generated_label(raw_text, processor, image_size)

                    true_5 = CLASS_NAME_TO_ID.get(true_label, -1)
                    pred_5 = CLASS_NAME_TO_ID.get(pred_label, -1)
                    if pred_5 < 0:
                        invalid_pred_count += 1
                    if true_5 == pred_5:
                        correct_5 += 1
                    for class_id in range(len(CLASS_NAMES)):
                        tp_5[class_id] += int(true_5 == class_id and pred_5 == class_id)
                        fp_5[class_id] += int(true_5 != class_id and pred_5 == class_id)
                        fn_5[class_id] += int(true_5 == class_id and pred_5 != class_id)

                    true_2 = class_to_binary_id(true_label)
                    pred_2 = class_to_binary_id(pred_label)
                    true_2 = true_2 if true_2 is not None else -1
                    pred_2 = pred_2 if pred_2 is not None else -1
                    if true_2 == pred_2:
                        correct_2 += 1
                    for class_id in range(2):
                        tp_2[class_id] += int(true_2 == class_id and pred_2 == class_id)
                        fp_2[class_id] += int(true_2 != class_id and pred_2 == class_id)
                        fn_2[class_id] += int(true_2 == class_id and pred_2 != class_id)

        stats = [val_loss_sum, val_item_count, correct_5, correct_2, invalid_pred_count]
        stats.extend(tp_5 + fp_5 + fn_5 + tp_2 + fp_2 + fn_2)
        stats_tensor = torch.tensor(stats, dtype=torch.float64, device=device)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        stats = stats_tensor.cpu().tolist()

        offset = 0
        total_val_loss = stats[offset]; offset += 1
        total_items = int(stats[offset]); offset += 1
        total_correct_5 = int(stats[offset]); offset += 1
        total_correct_2 = int(stats[offset]); offset += 1
        total_invalid = int(stats[offset]); offset += 1
        total_tp_5 = stats[offset:offset + len(CLASS_NAMES)]; offset += len(CLASS_NAMES)
        total_fp_5 = stats[offset:offset + len(CLASS_NAMES)]; offset += len(CLASS_NAMES)
        total_fn_5 = stats[offset:offset + len(CLASS_NAMES)]; offset += len(CLASS_NAMES)
        total_tp_2 = stats[offset:offset + 2]; offset += 2
        total_fp_2 = stats[offset:offset + 2]; offset += 2
        total_fn_2 = stats[offset:offset + 2]

        avg_val_loss = total_val_loss / max(1, total_items)
        val_5class_acc = total_correct_5 / max(1, total_items)
        val_2class_acc = total_correct_2 / max(1, total_items)
        val_5class_macro_f1 = sum(
            f1_from_counts(total_tp_5[i], total_fp_5[i], total_fn_5[i]) for i in range(len(CLASS_NAMES))
        ) / len(CLASS_NAMES)
        val_2class_macro_f1 = sum(
            f1_from_counts(total_tp_2[i], total_fp_2[i], total_fn_2[i]) for i in range(2)
        ) / 2
        final_metrics = {
            "val_loss": avg_val_loss,
            "val_5class_acc": val_5class_acc,
            "val_5class_macro_f1": val_5class_macro_f1,
            "val_2class_acc": val_2class_acc,
            "val_2class_macro_f1": val_2class_macro_f1,
            "invalid_pred_count": total_invalid,
            "val_item_count": total_items,
        }

        if rank == 0:
            print(f"Average Validation Loss: {avg_val_loss}")
            print(
                f"Validation Metrics on {val_name}: "
                f"5class_acc={val_5class_acc:.4f}, "
                f"5class_macro_f1={val_5class_macro_f1:.4f}, "
                f"2class_acc={val_2class_acc:.4f}, "
                f"2class_macro_f1={val_2class_macro_f1:.4f}, "
                f"invalid_pred={total_invalid}/{total_items}"
            )

    model.train()
    return final_metrics


def metric_selection_key(metrics):
    return (
        metrics["val_5class_macro_f1"],
        metrics["val_5class_acc"],
        metrics["val_2class_macro_f1"],
        metrics["val_2class_acc"],
        -metrics["val_loss"],
    )


def train_model(rank, world_size, dataset_name, batch_size=6, use_lora=False, epochs=15, lr=1e-6, eval_steps=1000, run_name=None, max_val_item_count=1000):
    setup(rank, world_size)
    print(f"[rank {rank}] → cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})")

    device = torch.device(f"cuda:{rank}")
    if run_name is None:
        run_name = "backbone_frozenVit_5class_moe_0.05edr_0.01cca"

    if rank == 0:
        print(f"[rank {rank}] train_model called with arguments:\n"
            f"  world_size={world_size}\n"
            f"  dataset_name={dataset_name!r}\n"
            f"  batch_size={batch_size}\n"
            f"  use_lora={use_lora}\n"
            f"  epochs={epochs}\n"
            f"  lr={lr}\n"
            f"  eval_steps={eval_steps}\n"
            f"  run_name={run_name!r}\n"
            f"  max_val_item_count={max_val_item_count}\n"
            f"  edr_loss_weight={EDR_LOSS_WEIGHT}\n"
            f"  cca_loss_weight={CCA_LOSS_WEIGHT}")

    if dataset_name == "docvqa":
        train_dataset = DocVQADataset(split='train')
        val_datasets = {"docvqa": DocVQADataset(split='validation')}
    elif dataset_name == "cauldron":
        train_dataset = TheCauldronDataset(split='train')
        val_datasets = {
            "cauldron": TheCauldronDataset(split='validation'),
            "docvqa": DocVQADataset(split='validation')
        }
    elif dataset_name == 'fakenews':
        train_csv_path_str = "/data1/hy/OmniFake_final_10K_with_visual/train_final_balanced_clean_with_bbox.csv"
        train_dataset = FakeDataset(
            csv_file_path=train_csv_path_str,
            image_directory_path="/data1/hy/OmniFake_final_10K_with_visual/"
        )
        val_datasets = {"fakenews": FakeDataset(
            csv_file_path="/data1/hy/OmniFake_final_10K_with_visual/val_final_balanced_clean_with_bbox.csv",
            image_directory_path="/data1/hy/OmniFake_final_10K_with_visual/"
        )}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model_root_path = "/data1/hy/huggingface_model/Florence-2-base-moe_edr_cca"

    model = AutoModelForCausalLM.from_pretrained(
        model_root_path, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_root_path, trust_remote_code=True
    )
    if hasattr(model, "language_model"):
        model.language_model.edr_loss_weight = EDR_LOSS_WEIGHT
        model.language_model.cca_loss_weight = CCA_LOSS_WEIGHT

    if rank == 0:
        print(f"[rank {rank}] >>> 前 2 条训练样本:")
        for i in range(min(2, len(train_dataset))):
            print(f"  [{i}]", train_dataset[i])
        for name, val_ds in val_datasets.items():
            print(f"[rank {rank}] >>> 验证集 '{name}' 的前 2 条样本:")
            for i in range(min(2, len(val_ds))):
                print(f"  [{i}]", val_ds[i])

    if use_lora:
        print("----------lora---------")
        TARGET_MODULES = [
            "q_proj", "o_proj", "k_proj", "v_proj",
            "linear", "Conv2d", "lm_head", "fc2"
        ]

        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, config)


    for param in model.vision_tower.parameters():
        param.requires_grad = False


    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {100*trainable/total:.4f}")

    model = DDP(model, device_ids=[rank])

    num_workers = 0
    train_loader, val_loaders = create_data_loaders(
        train_dataset,
        val_datasets,
        batch_size,
        num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    if rank == 0:
        run_root = f"/data1/hy/UMFDet_ckp/backbone_ckp/{run_name}"
        os.makedirs(run_root, exist_ok=True)
        log_path = os.path.join(run_root, "training.log")
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(f"Model_Path: {model_root_path}\n")
                f.write(f"Batch_Size: {batch_size}\n")
                f.write(f"Learning_Rate: {lr}\n")
                f.write(f"EDR_Loss_Weight: {EDR_LOSS_WEIGHT}\n")
                f.write(f"CCA_Loss_Weight: {CCA_LOSS_WEIGHT}\n")
                if 'train_csv_path_str' in locals():
                    f.write(f"Train_CSV_Path: {train_csv_path_str}\n")
                f.write(f"Prompt: {PROMPT_TEMPLATE}\n")
                f.write(
                    "epoch\ttrain_loss\tedr_loss\tcca_loss\taux_loss\tval_loss\t"
                    "val_5class_acc\tval_5class_macro_f1\tval_2class_acc\tval_2class_macro_f1\tbest_epoch\n"
                )

    best_epoch = -1
    best_metric_key = None

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_edr_loss = 0
        train_cca_loss = 0
        train_aux_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", position=rank
        ):
            inputs, answers = batch

            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)

            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)

            expert_labels = answers_to_expert_labels(answers, device)
            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels, expert_labels=expert_labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            if getattr(outputs, "edr_loss", None) is not None:
                train_edr_loss += outputs.edr_loss.detach().float().item()
            if getattr(outputs, "cca_loss", None) is not None:
                train_cca_loss += outputs.cca_loss.detach().float().item()
            if getattr(outputs, "aux_loss", None) is not None:
                train_aux_loss += outputs.aux_loss.detach().float().item()
            global_step += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_train_edr_loss = train_edr_loss / len(train_loader)
        avg_train_cca_loss = train_cca_loss / len(train_loader)
        avg_train_aux_loss = train_aux_loss / len(train_loader)

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.4f} "
                f"| EDR: {avg_train_edr_loss:.4f} "
                f"CCA: {avg_train_cca_loss:.4f} "
                f"AUX: {avg_train_aux_loss:.4f}"
            )

        val_metrics = evaluate_model(
            rank, world_size, model, val_loaders, device,
            train_loss, processor, global_step, batch_size, max_val_item_count
        )

        if rank == 0:
            current_metric_key = metric_selection_key(val_metrics)
            if best_metric_key is None or current_metric_key > best_metric_key:
                best_metric_key = current_metric_key
                best_epoch = epoch + 1

            with open(os.path.join(f"/data1/hy/UMFDet_ckp/backbone_ckp/{run_name}", "training.log"), "a") as f:
                f.write(
                    f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_train_edr_loss:.6f}\t"
                    f"{avg_train_cca_loss:.6f}\t{avg_train_aux_loss:.6f}\t"
                    f"{val_metrics['val_loss']:.6f}\t{val_metrics['val_5class_acc']:.6f}\t"
                    f"{val_metrics['val_5class_macro_f1']:.6f}\t{val_metrics['val_2class_acc']:.6f}\t"
                    f"{val_metrics['val_2class_macro_f1']:.6f}\t{best_epoch}\n"
                )

        if rank == 0:
            output_dir = f"/data1/hy/UMFDet_ckp/backbone_ckp/{run_name}/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train Florence-2 model on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["docvqa", "cauldron", "fakenews"], help="Dataset to train on")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size for training")
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=1000, help="Maximum number of items to evaluate on during validation")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(">>> visible gpus =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(">>> world_size =", torch.cuda.device_count())
    mp.spawn(
        train_model,
        args=(world_size, args.dataset, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, args.run_name, args.max_val_item_count),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
