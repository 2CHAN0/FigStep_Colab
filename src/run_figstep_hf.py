import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

from generate_prompts import QueryType, gen_query


@dataclass
class QuerySample:
    """Container for a SafeBench sample."""

    task_id: int
    category_name: str
    question: str
    instruction: str


def load_sample_from_dataset(dataset: Path, task_id: Optional[int], index: int) -> QuerySample:
    df = pd.read_csv(dataset)
    row = df[df["task_id"] == task_id].iloc[0] if task_id is not None else df.iloc[index]
    return QuerySample(
        task_id=int(row["task_id"]),
        category_name=str(row["category_name"]),
        question=str(row["question"]),
        instruction=str(row["instruction"]),
    )


def ensure_sample(args: argparse.Namespace) -> QuerySample:
    if args.question and args.instruction:
        return QuerySample(
            task_id=args.task_id or 0,
            category_name=args.category_name or "custom",
            question=args.question,
            instruction=args.instruction,
        )
    if not args.dataset:
        raise SystemExit("Either provide --dataset or both --question and --instruction.")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    return load_sample_from_dataset(dataset_path, args.task_id, args.index)


def save_prompt_and_image(
    output_dir: Path, sample: QuerySample, prompt: str, image: Optional[Image]
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{sample.task_id:03d}_{sample.category_name.replace(' ', '_')}"
    (output_dir / f"{slug}_prompt.txt").write_text(prompt, encoding="utf-8")
    if image is None:
        return None
    image_path = output_dir / f"{slug}_figstep.png"
    image.save(image_path)
    return image_path


def dtype_from_string(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    normalized = name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise SystemExit(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def resolve_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hf_model(
    model_id: str, device: str, torch_dtype: Optional[torch.dtype], load_in_4bit: bool
):
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if load_in_4bit:
        model_kwargs.update({"device_map": "auto", "load_in_4bit": True})
    else:
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
    except ValueError:
        # Fallback for models that only expose CausalLM heads (e.g., some Qwen checkpoints).
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not load_in_4bit:
        model.to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def generate_response(
    model,
    processor,
    prompt: str,
    image: Optional[Image],
    device: str,
    move_inputs: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = build_processor_inputs(processor, prompt, image)
    if not inputs:
        raise SystemExit("Failed to prepare processor inputs.")
    if move_inputs:
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FigStep prompts/images and query a Hugging Face Transformers VLM."
    )
    parser.add_argument(
        "--dataset",
        default="data/question/SafeBench-Tiny.csv",
        help="CSV file with SafeBench samples.",
    )
    parser.add_argument("--task-id", type=int, default=None, help="Specific SafeBench task_id.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index to use when --task-id is not provided (default: 0).",
    )
    parser.add_argument("--question", help="Custom question text.")
    parser.add_argument("--instruction", help="Custom instruction text.")
    parser.add_argument("--category-name", help="Category name for custom samples.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Hugging Face model id (default: Qwen/Qwen2-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Torch dtype for model weights (default: float16).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model in 4-bit quantization (requires bitsandbytes).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature (0 for greedy)."
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to store generated prompts and images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = ensure_sample(args)

    prompt, image = gen_query(QueryType.figstep, sample.question, sample.instruction)
    image_path = save_prompt_and_image(Path(args.output_dir), sample, prompt, image)
    if image_path:
        print(f"Saved FigStep image to: {image_path}")
    print(f"Using prompt:\n{prompt}\n")

    device = resolve_device(args.device)
    torch_dtype = None if args.load_in_4bit else dtype_from_string(args.torch_dtype)
    model, processor = load_hf_model(args.model_id, device, torch_dtype, args.load_in_4bit)

    response = generate_response(
        model=model,
        processor=processor,
        prompt=prompt,
        image=image,
        device=device,
        move_inputs=not args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("=== Hugging Face model response ===")
    print(response)


def build_processor_inputs(processor, prompt: str, image: Optional[Image]):
    """Create processor inputs with correct multimodal tokens/placeholders."""
    has_image = image is not None
    # Some processors (e.g., Qwen/Qwen2-VL) expect chat templates with explicit image blocks.
    if has_image and hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return processor(text=text, images=[image], return_tensors="pt")
    if has_image:
        # Fall back to inserting a generic placeholder if the prompt lacks one.
        placeholder = "<image>"
        text = prompt if placeholder in prompt else f"{placeholder}\n{prompt}"
        return processor(text=text, images=image, return_tensors="pt")
    return processor(text=prompt, return_tensors="pt")


if __name__ == "__main__":
    main()
