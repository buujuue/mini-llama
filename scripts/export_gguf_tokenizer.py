#!/usr/bin/env python3
# Copyright (c) 2026 yus3nable
# SPDX-License-Identifier: MIT
"""将 GGUF tokenizer metadata 导出为现有 BpeTokenizer 可加载的文件。

移植自 references/mini-llama-cpp/scripts/export_gguf_tokenizer.py。
仅依赖 Python 标准库；读取小端 GGUF v2/v3 的 metadata，不加载模型权重。
输出 vocab.json、merges.txt、special_tokens.json 和可选的 chat_template.txt。
"""

import argparse
import json
import struct
import sys
from pathlib import Path


VALUE_FORMATS = {
    0: "<B",  # uint8
    1: "<b",  # int8
    2: "<H",  # uint16
    3: "<h",  # int16
    4: "<I",  # uint32
    5: "<i",  # int32
    6: "<f",  # float32
    7: "<?",  # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
TOKEN_ID_KEYS = {
    "tokenizer.ggml.bos_token_id": "bos_id",
    "tokenizer.ggml.eos_token_id": "eos_id",
    "tokenizer.ggml.unknown_token_id": "unk_id",
    "tokenizer.ggml.padding_token_id": "pad_id",
}


def read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"GGUF 文件被截断：偏移 {f.tell()}，需要 {size} 字节，读到 {len(data)} 字节")
    return data


def read_scalar(f, fmt):
    return struct.unpack(fmt, read_exact(f, struct.calcsize(fmt)))[0]


def read_gguf_string(f):
    size = read_scalar(f, "<Q")
    return read_exact(f, size).decode("utf-8")


def skip_gguf_value(f, vtype):
    if vtype in VALUE_FORMATS:
        read_scalar(f, VALUE_FORMATS[vtype])
    elif vtype == 8:  # string
        read_gguf_string(f)
    elif vtype == 9:  # array
        etype = read_scalar(f, "<I")
        length = read_scalar(f, "<Q")
        for _ in range(length):
            skip_gguf_value(f, etype)
    else:
        raise ValueError(f"不支持的 GGUF metadata 类型：{vtype}")


def read_gguf_array(f, vtype, expected_type, key):
    if vtype != 9:
        raise ValueError(f"{key} 应为 GGUF 数组，实际类型为 {vtype}")
    etype = read_scalar(f, "<I")
    length = read_scalar(f, "<Q")
    if etype != expected_type:
        raise ValueError(f"{key} 的数组元素类型应为 {expected_type}，实际为 {etype}")
    if etype == 8:
        return [read_gguf_string(f) for _ in range(length)]
    return [read_scalar(f, VALUE_FORMATS[etype]) for _ in range(length)]


def read_gguf_metadata(path):
    meta = dict.fromkeys((
        "tokens", "merges", "token_types", "bos_id", "eos_id",
        "unk_id", "pad_id", "chat_template",
    ))
    with open(path, "rb") as f:
        if read_exact(f, 4) != b"GGUF":
            raise ValueError("GGUF 魔数不匹配：预期 GGUF")
        version = read_scalar(f, "<I")
        if version not in (2, 3):
            raise ValueError(f"不支持 GGUF 版本 {version}，仅支持小端 v2/v3")
        read_scalar(f, "<Q")  # tensor_count；只导出 metadata，不读取张量信息和权重。
        metadata_count = read_scalar(f, "<Q")

        for _ in range(metadata_count):
            key = read_gguf_string(f)
            vtype = read_scalar(f, "<I")
            if key == "tokenizer.ggml.tokens":
                meta["tokens"] = read_gguf_array(f, vtype, 8, key)
            elif key == "tokenizer.ggml.merges":
                meta["merges"] = read_gguf_array(f, vtype, 8, key)
            elif key == "tokenizer.ggml.token_type":
                meta["token_types"] = read_gguf_array(f, vtype, 5, key)
            elif key in TOKEN_ID_KEYS:
                if vtype != 4:
                    raise ValueError(f"{key} 应为 uint32，实际类型为 {vtype}")
                meta[TOKEN_ID_KEYS[key]] = read_scalar(f, "<I")
            elif key == "tokenizer.chat_template":
                if vtype != 8:
                    raise ValueError(f"{key} 应为字符串，实际类型为 {vtype}")
                meta["chat_template"] = read_gguf_string(f)
            else:
                skip_gguf_value(f, vtype)
    return meta


def export_tokenizer(gguf_path, out_dir):
    print(f"Reading tokenizer from {gguf_path}")
    meta = read_gguf_metadata(gguf_path)
    if not meta["tokens"]:
        raise ValueError("GGUF 缺少非空的 tokenizer.ggml.tokens 词表")
    vocab = {token: token_id for token_id, token in enumerate(meta["tokens"])}
    if len(vocab) != len(meta["tokens"]):
        raise ValueError("GGUF 词表含重复 token，无法导出为无歧义的 vocab.json")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  Wrote {len(vocab)} entries to {vocab_path}")

    # 数组顺序就是 merge rank；空数组也输出空文件，供 BpeTokenizer 加载。
    if meta["merges"] is not None:
        merges_path = out_dir / "merges.txt"
        with merges_path.open("w", encoding="utf-8", newline="\n") as f:
            for merge in meta["merges"]:
                f.write(merge + "\n")
        print(f"  Wrote {len(meta['merges'])} merges to {merges_path}")

    special = {
        key: meta[key] if meta[key] is not None else -1
        for key in TOKEN_ID_KEYS.values()
    }
    special_path = out_dir / "special_tokens.json"
    with special_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(special, f, indent=2)
        f.write("\n")
    print(f"  Wrote special tokens to {special_path}")

    if meta["chat_template"] is not None:
        template_path = out_dir / "chat_template.txt"
        with template_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(meta["chat_template"])
        print(f"  Wrote chat template to {template_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="导出 GGUF tokenizer，供 C++ BpeTokenizer 加载")
    parser.add_argument("gguf", help="输入的小端 GGUF v2/v3 文件")
    parser.add_argument("--out-dir", default="models/chat", help="输出目录（默认：models/chat）")
    args = parser.parse_args()
    try:
        export_tokenizer(args.gguf, args.out_dir)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
