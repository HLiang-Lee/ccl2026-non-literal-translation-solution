---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.5-9B-Base
tags:
- translation
- chinese-idioms
- ccl2026
---

# CCL2026 中文非字面翻译评测模型

本仓库为 **CCL2026** 相关评测任务所使用的模型权重与配置文件，采用 Hugging Face Transformers 格式保存，兼容 Transformers、vLLM、SGLang 等主流推理框架。

模型基于 **Qwen3.5-9B-Base** 进行有监督微调（SFT），面向中文习语、谚语及文化负载表达的翻译与判别场景，支持以下两类任务：

- **Task 1（翻译生成）**：将中文习语/谚语翻译为自然、地道的英文，优先复用对应的英文习语或谚语，否则给出清晰的意译。
- **Task 2（候选判别）**：给定一个中文表达及若干英文候选译文，对每个候选独立判别为 `gold`（应被选中的等价译法）或 `silver`（解释性意译或非等价译法）。

## Model Overview

- Type: Causal Language Model
- Base Model: Qwen/Qwen3.5-9B-Base
- Training Stage: Supervised Fine-Tuning (SFT)
- Number of Parameters: 9B
- Context Length: 参见 `config.json`
- Precision: bfloat16

## 使用方式

推理约定（与训练时的提示词保持一致）：

- **Task 1** system 提示引导模型输出地道英文翻译；user 输入格式为 `Chinese: {表达}\nEnglish translation:`。
- **Task 2** system 提示引导模型逐项判别；user 输入包含中文表达与按字母编号的候选列表，模型对每个选项输出一行 `LETTER: gold` 或 `LETTER: silver`。
- 解码建议使用贪心（`temperature=0`），并关闭思考模式（`enable_thinking=False`）。

配套推理脚本见 `infer.py`，依赖版本见 `requirements.txt`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "path/to/this/repo"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True,
).eval()

messages = [
    {"role": "system", "content": "You are an expert translator ..."},
    {"role": "user", "content": "Chinese: 入乡随俗\nEnglish translation:"},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## License

本模型继承基座模型 Qwen3.5-9B-Base 的许可协议（Apache-2.0），详见 [LICENSE](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE)。
