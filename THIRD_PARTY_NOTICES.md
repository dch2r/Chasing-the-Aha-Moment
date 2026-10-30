# Sources and license scope

Project owner: Diwesh Chaurasia. The root MIT license covers project-owned additions; it does not replace the terms of upstream notebooks, model files or dataset excerpts.

| Material | Source | Notice |
| --- | --- | --- |
| Historical Llama notebook templates in `legacy/` | [Unsloth notebooks](https://github.com/unslothai/notebooks) | LGPL-3.0 notices retained; license texts in `LICENSES/`. These are adapted notebooks, not an independently invented GRPO implementation. |
| Historical reward and data-preparation example | [William Brown's GRPO example](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) | Existing attribution retained in the notebooks. |
| Base model and tokenizer | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache-2.0. The bundled LoRA adapter is a fine-tuning artifact, not the base model. Tokenizer files originate from Qwen; license text included. |
| GSM8K question excerpts and answers | [OpenAI grade-school-math](https://github.com/openai/grade-school-math) | MIT; original copyright and license in `LICENSES/GSM8K-MIT.txt`. |
| Training/runtime libraries | PyTorch, Hugging Face Transformers, TRL, PEFT, Accelerate, datasets, bitsandbytes | Installed separately and retain their respective licenses. |

The saved adapter changes model behavior through rank-16 LoRA training; it does not modify the bundled tokenizer. Project utilities and revisions were developed with AI assistance and validated with tests and recorded Colab runs.

Edits to historical material: the license attribution was updated and two stale repository badges were replaced with local-upload instructions. Notebook training code and recorded outputs were not changed. Dataset character names are preserved as recorded data.
