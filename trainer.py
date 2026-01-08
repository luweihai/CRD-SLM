import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import shutil
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from config import LABEL_MAP, PROMPT_TEMPLATE, NUM_LABELS
from model import MultiTaskMoLoRAModel
from utils import save_checkpoint, load_checkpoint

def evaluate_and_generate(model, tokenizer, test_dataset_cls, args, device):
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset_cls), args.batch_size), desc="正在评估分类性能"):
            batch_data = test_dataset_cls[i:i + args.batch_size]
            if not batch_data: continue
            contents = [entry['content'] for entry in batch_data]
            true_label_ids = [LABEL_MAP[entry['label']] for entry in batch_data]
            inputs = tokenizer(contents, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(device)
            
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], task='classifier')
            logits = outputs['logits']
            predicted_label_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(predicted_label_ids)
            all_labels.extend(true_label_ids)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    generation_examples = []
    sample_for_generation = random.sample(test_dataset_cls, k=min(len(test_dataset_cls), 5))
    print("\n--- 正在生成Reasoning样本（定性评估）---")
    
    unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
    
    for entry in sample_for_generation:
        prompt = PROMPT_TEMPLATE.format(content=entry['content'])
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(device)
        
        with torch.no_grad():
            unwrapped_model.backbone.set_adapter(["reasoner", "shared"])
            generated_ids = unwrapped_model.backbone.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                num_beams=3,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generation_examples.append({
            "content": entry['content'],
            "ground_truth_label": entry['label'],
            "generated_reason": generated_text.strip()
        })
        print(f"新闻 (截断): {entry['content'][:100]}...")
        print(f"生成理由: {generated_text.strip()}\n")
    
    generation_examples_path = os.path.join(args.output_dir, "generation_examples.json")
    with open(generation_examples_path, 'w', encoding='utf-8') as f:
        json.dump(generation_examples, f, ensure_ascii=False, indent=4)

    return {"accuracy": accuracy, "f1_score": f1}


def train(train_cls_data, train_dpo_data, test_dataset_cls, args, device_ids, device):
    
    latest_checkpoint_dir = os.path.join(args.output_dir, "latest_checkpoint")
    best_checkpoint_dir = os.path.join(args.output_dir, "best_checkpoint")
    test_results_log_path = os.path.join(args.output_dir, "test_results_log.json")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MultiTaskMoLoRAModel(args.model_path, NUM_LABELS, dpo_beta=args.dpo_beta).to(device)
    unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
    unwrapped_model.reference_model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    classification_loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    
    start_epoch, step_count = load_checkpoint(model, optimizer, latest_checkpoint_dir, device)

    if torch.cuda.is_available() and len(device_ids) > 1:
        print(f"使用 {len(device_ids)} 个 GPUs 进行数据并行训练...")
        model = nn.DataParallel(model, device_ids=device_ids)

    if os.path.exists(test_results_log_path):
        try:
            with open(test_results_log_path, 'r', encoding='utf-8') as f:
                test_results_log = json.load(f)
        except json.JSONDecodeError:
            test_results_log = []
    else:
        test_results_log = []
        
    best_f1 = max((res.get('f1_score', 0) for res in test_results_log), default=0.0)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- 第 {epoch+1}/{args.epochs} 轮 ---")
        np.random.shuffle(train_cls_data)
        np.random.shuffle(train_dpo_data)
        model.train()
        
        cls_batches = [train_cls_data[i:i+args.batch_size] for i in range(0, len(train_cls_data), args.batch_size)]
        dpo_batches = [train_dpo_data[i:i+args.batch_size] for i in range(0, len(train_dpo_data), args.batch_size)]
        
        num_batches = max(len(cls_batches), len(dpo_batches))
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for i in pbar:
            optimizer.zero_grad()
            
            cls_loss, dpo_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            if cls_batches:
                cls_batch = cls_batches[i % len(cls_batches)]
                contents = [entry['content'] for entry in cls_batch]
                true_labels = torch.tensor([LABEL_MAP[entry['label']] for entry in cls_batch], device=device)
                inputs_cls = tokenizer(contents, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(device)
                
                outputs_cls = model(input_ids=inputs_cls.input_ids, attention_mask=inputs_cls.attention_mask, task='classifier')
                cls_logits = outputs_cls['logits']
                cls_loss = classification_loss_fn(cls_logits, true_labels)

            if dpo_batches:
                dpo_batch_data = dpo_batches[i % len(dpo_batches)]
                
                prompts = [PROMPT_TEMPLATE.format(content=entry['content']) for entry in dpo_batch_data]
                chosen_responses = [entry['reasoning_text'] + tokenizer.eos_token for entry in dpo_batch_data]
                rejected_responses_1 = [entry['contentneg1'] + tokenizer.eos_token for entry in dpo_batch_data]
                rejected_responses_2 = [entry['contentneg2'] + tokenizer.eos_token for entry in dpo_batch_data]
                rejected_responses_3 = [entry['contentneg3'] + tokenizer.eos_token for entry in dpo_batch_data]
                
                prompt_tokens = tokenizer(prompts, padding=False, truncation=True, max_length=args.max_length)
                prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
                
                chosen_full = [p + r for p, r in zip(prompts, chosen_responses)]
                rejected_full_1 = [p + r for p, r in zip(prompts, rejected_responses_1)]
                rejected_full_2 = [p + r for p, r in zip(prompts, rejected_responses_2)]
                rejected_full_3 = [p + r for p, r in zip(prompts, rejected_responses_3)]

                all_sequences = chosen_full + rejected_full_1 + rejected_full_2 + rejected_full_3
                all_tokens = tokenizer(all_sequences, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
                
                bs = len(dpo_batch_data)
                dpo_input_batch = {
                    "prompt_lengths": torch.tensor(prompt_lengths, device=device),
                    "chosen_input_ids": all_tokens.input_ids[:bs].to(device),
                    "chosen_attention_mask": all_tokens.attention_mask[:bs].to(device),
                    "rejected_input_ids_1": all_tokens.input_ids[bs:2*bs].to(device),
                    "rejected_attention_mask_1": all_tokens.attention_mask[bs:2*bs].to(device),
                    "rejected_input_ids_2": all_tokens.input_ids[2*bs:3*bs].to(device),
                    "rejected_attention_mask_2": all_tokens.attention_mask[2*bs:3*bs].to(device),
                    "rejected_input_ids_3": all_tokens.input_ids[3*bs:].to(device),
                    "rejected_attention_mask_3": all_tokens.attention_mask[3*bs:].to(device),
                }

                outputs_dpo = model(task='reasoner', dpo_batch=dpo_input_batch)
                dpo_loss = outputs_dpo['loss']

            total_loss = cls_loss + dpo_loss

            if isinstance(model, nn.DataParallel):
                total_loss = total_loss.mean()
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "cls_loss": f"{cls_loss.mean().item():.3f}", 
                "dpo_loss": f"{dpo_loss.mean().item():.3f}",
            })
            step_count += 1

        print(f"\n--- 完成第 {epoch+1} 轮，开始评估 ---")
        metrics = evaluate_and_generate(model, tokenizer, test_dataset_cls, args, device)
        print(f"Epoch {epoch+1} 评估结果: Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")

        epoch_result = {"epoch": epoch + 1, "step": step_count, **metrics}
        test_results_log.append(epoch_result)
        with open(test_results_log_path, 'w', encoding='utf-8') as f:
            json.dump(test_results_log, f, indent=4)

        save_checkpoint(model, optimizer, epoch, step_count, latest_checkpoint_dir)
        if metrics['f1_score'] > best_f1:
            print(f"发现新的最佳模型 (F1: {metrics['f1_score']:.4f})，保存到 '{os.path.basename(best_checkpoint_dir)}'...")
            best_f1 = metrics['f1_score']
            if os.path.exists(best_checkpoint_dir):
                shutil.rmtree(best_checkpoint_dir)
            save_checkpoint(model, optimizer, epoch, step_count, best_checkpoint_dir)

    print("\n所有训练轮次完成。")
