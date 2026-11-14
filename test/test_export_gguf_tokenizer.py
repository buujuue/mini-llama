"""使用自建 GGUF metadata 验证导出 CLI，无需模型或第三方依赖。"""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_gguf_tokenizer.py"


def gguf_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def metadata(key, vtype, value):
    return gguf_string(key) + struct.pack("<I", vtype) + value


def string_array(values):
    return struct.pack("<IQ", 8, len(values)) + b"".join(map(gguf_string, values))


def gguf(entries, version=3):
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(entries)) + b"".join(entries)


class ExportGgufTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "tokenizer.gguf"
        self.output = Path(self.temp.name) / "exported"

    def run_export(self, data):
        self.source.write_bytes(data)
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.source), "--out-dir", str(self.output)],
            capture_output=True, encoding="utf-8",
        )

    def test_exports_unicode_ids_rank_order_and_template(self):
        tokens = ["h", "i", "hi", "你好", "<|im_end|>", '"\\\n']
        merges = ["h i", "Ġ hi"]
        template = "{{ messages[0]['content'] }}\n你好\n"
        entries = [
            # 不相关 metadata 必须正确跳过，避免后续字段错位。
            metadata("general.name", 8, gguf_string("tiny")),
            metadata("general.alignment", 4, struct.pack("<I", 32)),
            metadata("ignored.array", 9, struct.pack("<IQff", 6, 2, 0.5, 1.0)),
            metadata("tokenizer.ggml.tokens", 9, string_array(tokens)),
            metadata("tokenizer.ggml.merges", 9, string_array(merges)),
            metadata("tokenizer.ggml.token_type", 9,
                     struct.pack("<IQ6i", 5, 6, 1, 1, 1, 1, 3, 1)),
            metadata("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
            metadata("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 4)),
            metadata("tokenizer.chat_template", 8, gguf_string(template)),
        ]
        result = self.run_export(gguf(entries))
        self.assertEqual(result.returncode, 0, result.stderr)
        vocab = json.loads((self.output / "vocab.json").read_text(encoding="utf-8"))
        self.assertEqual(vocab, dict(zip(tokens, range(len(tokens)))))
        self.assertEqual((self.output / "merges.txt").read_bytes(),
                         ("\n".join(merges) + "\n").encode("utf-8"))
        special = json.loads((self.output / "special_tokens.json").read_text())
        self.assertEqual(special, {"bos_id": 0, "eos_id": 4, "unk_id": -1, "pad_id": -1})
        self.assertEqual((self.output / "chat_template.txt").read_bytes(), template.encode("utf-8"))

    def test_v2_empty_merges_and_missing_optional_fields(self):
        result = self.run_export(gguf([
            metadata("tokenizer.ggml.tokens", 9, string_array(["a"])),
            metadata("tokenizer.ggml.merges", 9, string_array([])),
        ], version=2))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.output / "merges.txt").read_bytes(), b"")
        self.assertFalse((self.output / "chat_template.txt").exists())

    def test_rejects_invalid_header_and_version(self):
        for data in (b"NOPE" + gguf([])[4:], gguf([], version=1)):
            with self.subTest(data=data):
                result = self.run_export(data)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Error:", result.stderr)
                self.assertFalse(self.output.exists())

    def test_rejects_truncated_metadata(self):
        data = gguf([metadata("tokenizer.ggml.tokens", 9, string_array(["你好"]))])
        result = self.run_export(data[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("截断", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_array_element_type(self):
        result = self.run_export(gguf([
            metadata("tokenizer.ggml.tokens", 9, struct.pack("<IQi", 5, 1, 42)),
        ]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("数组元素类型", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_missing_or_duplicate_vocabulary(self):
        for entries in ([], [metadata("tokenizer.ggml.tokens", 9, string_array(["a", "a"]))]):
            with self.subTest(entries=entries):
                result = self.run_export(gguf(entries))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
