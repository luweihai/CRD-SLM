import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType
from typing import List, Dict

from utils import CustomPeftMixedModel 

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps_list: List[torch.FloatTensor],
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps_list: List[torch.FloatTensor],
    ) -> torch.FloatTensor:
        total_loss = 0.0
        
        for i in range(len(policy_rejected_logps_list)):
            policy_rejected_logps = policy_rejected_logps_list[i]
            reference_rejected_logps = reference_rejected_logps_list[i]

            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            
            logits = pi_logratios - ref_logratios
            
            loss = -F.logsigmoid(self.beta * logits)
            total_loss += loss.mean()
            
        return total_loss / len(policy_rejected_logps_list)


class MultiTaskMoLoRAModel(nn.Module):
    def __init__(self, model_name, num_labels, dpo_beta=0.1):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

        self.hidden_size = base_model.config.hidden_size
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        classifier_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.1, target_modules=target_modules, bias="none")
        reasoner_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.1, target_modules=target_modules, bias="none")
        shared_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.1, target_modules=target_modules, bias="none")

        self.backbone = CustomPeftMixedModel(base_model, classifier_lora_config, adapter_name="classifier")
        self.backbone.add_adapter("reasoner", reasoner_lora_config)
        self.backbone.add_adapter("shared", shared_lora_config)
        
        self.classification_head = nn.Linear(self.hidden_size * 2, num_labels)
        self.dpo_loss_fn = DPOLoss(beta=dpo_beta)
        
        print("多任务LoRA模型(含DPO)初始化完成，包含的适配器:", list(self.backbone.peft_config.keys()))

    def _get_log_probs(self, model, input_ids, attention_mask, prompt_lengths):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:].clone()
        token_log_probs = log_probs[:, :-1, :].gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            start_index = prompt_lengths[i] - 1
            end_index = attention_mask[i, 1:].sum()
            mask[i, start_index:end_index] = True
            
        masked_log_probs = token_log_probs * mask
        return masked_log_probs.sum(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, task: str = None, dpo_batch: Dict = None):
        if task not in ["classifier", "reasoner"]:
            raise ValueError("任务(task)参数必须是 'classifier' 或 'reasoner'")

        if task == "classifier":
            self.backbone.set_adapter(["classifier", "shared"])
            outputs_cls = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_state_cls = outputs_cls.hidden_states[-1]
            mean_pooled_vector = self.get_mean_pooled_vector(hidden_state_cls, attention_mask)

            self.backbone.set_adapter(["reasoner", "shared"])
            outputs_rea = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_state_rea = outputs_rea.hidden_states[-1]
            last_token_hidden_state = self.get_last_token_vector(hidden_state_rea, attention_mask)
            
            combined_features = torch.cat([mean_pooled_vector, last_token_hidden_state], dim=1)
            logits = self.classification_head(combined_features)
            
            return {"logits": logits}
        
        elif task == "reasoner":
            self.backbone.set_adapter(["reasoner", "shared"])
            policy_chosen_logps = self._get_log_probs(self.backbone, dpo_batch['chosen_input_ids'], dpo_batch['chosen_attention_mask'], dpo_batch['prompt_lengths'])
            policy_rejected_logps_list = [
                self._get_log_probs(self.backbone, dpo_batch['rejected_input_ids_1'], dpo_batch['rejected_attention_mask_1'], dpo_batch['prompt_lengths']),
                self._get_log_probs(self.backbone, dpo_batch['rejected_input_ids_2'], dpo_batch['rejected_attention_mask_2'], dpo_batch['prompt_lengths']),
                self._get_log_probs(self.backbone, dpo_batch['rejected_input_ids_3'], dpo_batch['rejected_attention_mask_3'], dpo_batch['prompt_lengths']),
            ]
            
            with torch.no_grad():
                reference_chosen_logps = self._get_log_probs(self.reference_model, dpo_batch['chosen_input_ids'], dpo_batch['chosen_attention_mask'], dpo_batch['prompt_lengths'])
                reference_rejected_logps_list = [
                    self._get_log_probs(self.reference_model, dpo_batch['rejected_input_ids_1'], dpo_batch['rejected_attention_mask_1'], dpo_batch['prompt_lengths']),
                    self._get_log_probs(self.reference_model, dpo_batch['rejected_input_ids_2'], dpo_batch['rejected_attention_mask_2'], dpo_batch['prompt_lengths']),
                    self._get_log_probs(self.reference_model, dpo_batch['rejected_input_ids_3'], dpo_batch['rejected_attention_mask_3'], dpo_batch['prompt_lengths']),
                ]

            dpo_loss = self.dpo_loss_fn(policy_chosen_logps, policy_rejected_logps_list, reference_chosen_logps, reference_rejected_logps_list)
            
            return {"loss": dpo_loss}

    def get_mean_pooled_vector(self, hidden_state, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden_state).float()
        masked_hidden_states = hidden_state * expanded_mask
        summed_hidden_states = masked_hidden_states.sum(dim=1)
        num_tokens = expanded_mask.sum(dim=1)
        num_tokens[num_tokens == 0] = 1e-9
        return summed_hidden_states / num_tokens

    def get_last_token_vector(self, hidden_state, attention_mask):
        batch_size = hidden_state.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]

