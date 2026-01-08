import sys
import torch
import argparse

from data_loader import load_and_split_dataset
from trainer import train

def main(args):
    sys.stdout.reconfigure(encoding='utf-8')

    if args.device_ids:
        device_ids = [int(i) for i in args.device_ids.split(',') if i.strip()]
    else:
        device_ids = []

    if torch.cuda.is_available() and device_ids:
        torch.cuda.set_device(f"cuda:{device_ids[0]}")
        device = f"cuda:{device_ids[0]}"
    else:
        device = "cpu"
    print(f"主设备设置为: {device}")

    print("正在加载和划分数据集...")
    train_cls_data, train_dpo_data = load_and_split_dataset(args.train_data_path)
    test_cls_data, _ = load_and_split_dataset(args.test_data_path)
    
    if not train_dpo_data:
        print("\n警告: 训练数据中没有可用于DPO-Reasoning任务的样本。")
    if not train_cls_data:
        print("\n警告: 训练数据中没有可用于分类任务的样本。")
    if not train_dpo_data and not train_cls_data:
        print("\n错误: 训练数据为空。")
        sys.exit(1)
        
    train(train_cls_data, train_dpo_data, test_cls_data, args, device_ids, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Task LoRA Model Training with DPO")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the local pretrained model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints and results.')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training JSON file with DPO samples.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the testing JSON file.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training. DPO requires more memory.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum token length for tokenizer.')
    parser.add_argument('--dpo_beta', type=float, default=0.1, help='Beta hyperparameter for DPO loss.')
    parser.add_argument('--device_ids', type=str, default='0', help='Comma-separated list of GPU device IDs to use. E.g., "0,1,2"')

    args = parser.parse_args()
    main(args)
