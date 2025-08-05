## Fine-tuning Qwen2-VL
In order to finetune qwen2-vl, you can follow the instructions from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

* First, update the data/dataset_info.json file with details about the data for fine-tuning.
* Next, execute the fine-tuning command.
```bash
llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
```
* Finally, merge the model's parameters for inference
```bash
llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml
```
