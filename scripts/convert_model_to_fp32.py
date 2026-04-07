import argparse
import os
import json
import safetensors
import safetensors.torch
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer_name = "tokenizer.json"
    shutil.copy(
        os.path.join(args.snap_dir, tokenizer_name),
        os.path.join(args.out_dir, tokenizer_name),
    )

    config_name = "config.json"
    with open(os.path.join(args.snap_dir, config_name), "r", encoding="utf-8") as f:
        configs = json.load(f)
        configs["torch_dtype"] = "float32"
        with open(
            os.path.join(args.out_dir, config_name), "w", encoding="utf-8"
        ) as o_f:
            json.dump(configs, o_f, ensure_ascii=False, indent=4)

    model_name = "model.safetensors"
    with safetensors.safe_open(
        os.path.join(args.snap_dir, model_name), framework="pt"
    ) as f:
        safetensors.torch.save_file(
            {k: f.get_tensor(k).float() for k in f.keys()},
            os.path.join(args.out_dir, model_name),
        )


if __name__ == "__main__":
    main()
