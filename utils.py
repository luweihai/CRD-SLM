import os
import shutil
import torch
import torch.nn as nn
from peft import PeftMixedModel, PeftConfig
from typing import Optional, Any


class CustomPeftMixedModel(PeftMixedModel):
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        if os.path.isfile(save_directory):
            raise ValueError(f"save_pretrained received a file path ({save_directory}), should be a directory.")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            missing = [ad for ad in selected_adapters if ad not in self.peft_config]
            if missing:
                raise ValueError(f"Cannot save these adapters {missing}, as they are not found in the current model. Available adapters: {list(self.peft_config.keys())}")

        os.makedirs(save_directory, exist_ok=True)
        
        for adapter_name in selected_adapters:
            adapter_config = self.peft_config[adapter_name]
            adapter_save_dir = os.path.join(save_directory, adapter_name)
            os.makedirs(adapter_save_dir, exist_ok=True)
            adapter_config.save_pretrained(adapter_save_dir)

            state_dict = self.state_dict()
            adapter_state_dict = {k: v for k, v in state_dict.items() if adapter_name in k}
            
            if safe_serialization:
                from safetensors.torch import save_file as safe_save_file
                safe_save_file(adapter_state_dict, os.path.join(adapter_save_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
            else:
                torch.save(adapter_state_dict, os.path.join(adapter_save_dir, "adapter_model.bin"))

        print(f"Adapters {selected_adapters} saved to subdirectories in {save_directory}")

    @classmethod
    def from_pretrained(cls, base_model, model_id: str, **kwargs):
        adapter_dirs = [d for d in os.listdir(model_id) if os.path.isdir(os.path.join(model_id, d))]
        if not adapter_dirs:
            raise ValueError(f"No adapter subdirectories found in {model_id}")

        print(f"Found adapters to load: {adapter_dirs}")

        first_adapter_name = adapter_dirs[0]
        first_adapter_config = PeftConfig.from_pretrained(os.path.join(model_id, first_adapter_name))
        model = cls(base_model, first_adapter_config, adapter_name=first_adapter_name)
        
        for adapter_name in adapter_dirs[1:]:
            adapter_config = PeftConfig.from_pretrained(os.path.join(model_id, adapter_name))
            model.add_adapter(adapter_name, adapter_config)
            
        for adapter_name in adapter_dirs:
            adapter_path = os.path.join(model_id, adapter_name)
            try:
                from safetensors.torch import load_file as safe_load_file
                adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                adapter_weights = safe_load_file(adapter_weights_path, device="cpu")
            except (ImportError, FileNotFoundError):
                 adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
                 adapter_weights = torch.load(adapter_weights_path, map_location="cpu")
            
            model.load_state_dict(adapter_weights, strict=False)

        print(f"All adapters ({adapter_dirs}) loaded successfully.")
        return model


def save_checkpoint(model, optimizer, epoch, step, path):
    os.makedirs(path, exist_ok=True)
    unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
    
    unwrapped_model.backbone.save_pretrained(path)
    torch.save(unwrapped_model.classification_head.state_dict(), os.path.join(path, "classification_head.pth"))

    state = {'epoch': epoch, 'step': step, 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, os.path.join(path, "trainer_state.pth"))
    print(f"检查点已完整保存到目录: {path}")


def load_checkpoint(model, optimizer, path, device):
    trainer_state_path = os.path.join(path, "trainer_state.pth")
    if not os.path.exists(trainer_state_path):
        print("未找到检查点目录或trainer state文件，将从头开始训练。")
        return 0, 0
    
    unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
    
    print(f"从 {path} 加载PEFT模型...")
    base_model = unwrapped_model.backbone.get_base_model()
    unwrapped_model.backbone = CustomPeftMixedModel.from_pretrained(base_model, path)
    
    head_path = os.path.join(path, "classification_head.pth")
    unwrapped_model.classification_head.load_state_dict(torch.load(head_path, map_location=device))
    
    state = torch.load(trainer_state_path, map_location=device)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    start_epoch = state.get('epoch', -1) + 1
    start_step = state.get('step', 0)
    
    print(f"成功从 {path} 加载检查点。将从 epoch {start_epoch} 继续。")
    print("已加载的适配器(专家):", list(unwrapped_model.backbone.peft_config.keys()))
    return start_epoch, start_step
