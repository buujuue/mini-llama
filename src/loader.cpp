#include "loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static Model LoadFailure(const std::string& message) {
    Model model;
    model.loaded = false;
    model.load_error = message;
    std::cerr << "Failed to load model: " << message << std::endl;
    return model;
}

static std::string InputContext(const std::string& content, size_t pos) {
    std::string context = "，当前输入中的字节位置: " + std::to_string(pos);
    if (pos >= content.size()) {
        return context + "，实际: <输入结束>";
    }
    const std::string preview = content.substr(pos, 48);
    std::string escaped;
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : preview) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    escaped += "\\x";
                    escaped += hex[c >> 4];
                    escaped += hex[c & 0xf];
                } else {
                    escaped += static_cast<char>(c);
                }
        }
    }
    if (content.size() - pos > preview.size()) escaped += "...";
    return context + "，附近内容: \"" + escaped + "\"";
}

static std::string FloatToString(float value) {
    std::ostringstream output;
    output.precision(std::numeric_limits<float>::max_digits10);
    output << value;
    return output.str();
}

static size_t SkipWhitespace(const std::string& content, size_t pos) {
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    return pos;
}

static size_t FindOptionalKey(const std::string& content, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    const size_t colon_pos = SkipWhitespace(content, key_pos + marker.size());
    if (colon_pos >= content.size() || content[colon_pos] != ':') {
        throw std::runtime_error("字段后缺少冒号 ':'，字段: " + key
            + InputContext(content, colon_pos));
    }
    const size_t value_pos = SkipWhitespace(content, colon_pos + 1);
    if (value_pos >= content.size()) {
        throw std::runtime_error("字段缺少值，字段: " + key
            + InputContext(content, value_pos));
    }
    return value_pos;
}

static size_t FindConfigKey(const std::string& content, const std::string& key) {
    const size_t pos = FindOptionalKey(content, key);
    if (pos == std::string::npos) {
        throw std::runtime_error("缺少必需字段: " + key
            + "，当前 JSON 对象必须包含字段 \"" + key + "\""
            + "，输入大小: " + std::to_string(content.size()) + " 字节");
    }
    return pos;
}

static void CheckNumberTerminator(const std::string& content, size_t value_pos,
                           size_t parsed, const std::string& key) {
    const size_t pos = SkipWhitespace(content, value_pos + parsed);
    if (pos < content.size() && content[pos] != ',' && content[pos] != '}' &&
        content[pos] != ']') {
        throw std::runtime_error("字段数值格式无效，字段: " + key
            + "，数字后应为 ','、'}' 或 ']'"
            + "，数值起始字节位置: " + std::to_string(value_pos)
            + InputContext(content, pos));
    }
}

static int ParseRequiredInt(const std::string& content, const std::string& key) {
    const size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("字段整数值无效，字段: " + key
            + "，预期整数范围: [" + std::to_string(std::numeric_limits<int>::min())
            + ", " + std::to_string(std::numeric_limits<int>::max()) + "]"
            + InputContext(content, value_pos));
    }
    CheckNumberTerminator(content, value_pos, parsed, key);
    return value;
}

static size_t ParseRequiredDimension(const std::string& content, const std::string& key) {
    const int value = ParseRequiredInt(content, key);
    if (value <= 0) {
        throw std::runtime_error("模型维度必须为正数，字段: " + key
            + "，实际值: " + std::to_string(value)
            + "，有效范围: [1, "
            + std::to_string(std::numeric_limits<int>::max()) + "]");
    }
    return static_cast<size_t>(value);
}

static std::string TrimWs(const std::string& text) {
    const size_t start = text.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    const size_t end = text.find_last_not_of(" \t\n\r");
    return text.substr(start, end - start + 1);
}

static int ParseStrictIntToken(const std::string& token, const std::string& field) {
    const std::string trimmed = TrimWs(token);
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(trimmed, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("整数值无效，字段: " + field
            + "，预期整数范围: [" + std::to_string(std::numeric_limits<int>::min())
            + ", " + std::to_string(std::numeric_limits<int>::max()) + "]"
            + InputContext(trimmed, 0));
    }
    if (parsed != trimmed.size()) {
        throw std::runtime_error("整数值无效，字段: " + field
            + "，预期整个字段均为整数"
            + "，已解析字符数: " + std::to_string(parsed)
            + InputContext(trimmed, 0));
    }
    return value;
}

static size_t ParseStrictSizeValue(const std::string& content, size_t value_pos,
                            const std::string& field) {
    value_pos = SkipWhitespace(content, value_pos);
    if (value_pos >= content.size() ||
        !std::isdigit(static_cast<unsigned char>(content[value_pos]))) {
        throw std::runtime_error("非负整数值无效，字段: " + field
            + "，数值必须以十进制数字开头"
            + InputContext(content, value_pos));
    }
    size_t parsed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("非负整数值无效，字段: " + field
            + "，有效范围: [0, "
            + std::to_string(std::numeric_limits<size_t>::max()) + "]"
            + InputContext(content, value_pos));
    }
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("非负整数值无效，字段: " + field
            + "，实际值: " + std::to_string(value)
            + "，有效范围: [0, "
            + std::to_string(std::numeric_limits<size_t>::max()) + "]"
            + InputContext(content, value_pos));
    }
    CheckNumberTerminator(content, value_pos, parsed, field);
    return static_cast<size_t>(value);
}

static float ParseRequiredFloat(const std::string& content, const std::string& key) {
    const size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    float value = 0.0f;
    try {
        value = std::stof(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("字段浮点数值无效，字段: " + key
            + "，预期可用 float 表示的数值"
            + InputContext(content, value_pos));
    }
    CheckNumberTerminator(content, value_pos, parsed, key);
    return value;
}

static std::string ParseStringValue(const std::string& content, size_t value_pos,
                              const std::string& key) {
    if (content[value_pos] != '"') {
        throw std::runtime_error("字段必须为字符串，字段: " + key
            + "，预期以双引号 '\"' 开头" + InputContext(content, value_pos));
    }
    std::string value;
    for (size_t pos = value_pos + 1; pos < content.size(); ++pos) {
        char c = content[pos];
        if (c == '"') {
            return value;
        }
        if (c == '\\') {
            if (++pos >= content.size()) {
                break;
            }
            c = content[pos];
            switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default:
                    throw std::runtime_error("字符串包含不支持的转义字符，字段: " + key
                        + "，反斜杠后支持的字符: \" \\ / b f n r t"
                        + InputContext(content, pos - 1));
            }
        } else if (static_cast<unsigned char>(c) < 0x20) {
            throw std::runtime_error("字符串值无效，字段: " + key
                + "，未转义的控制字符编码: "
                + std::to_string(static_cast<unsigned char>(c))
                + InputContext(content, pos));
        }
        value += c;
    }
    throw std::runtime_error("字符串未闭合，字段: " + key
        + "，输入结束前缺少闭合双引号 '\"'"
        + "，输入大小: " + std::to_string(content.size()) + " 字节"
        + InputContext(content, value_pos));
}

static std::string ParseRequiredStringField(const std::string& content,
                                      const std::string& key) {
    return ParseStringValue(content, FindConfigKey(content, key), key);
}

static std::string ParseOptionalStringField(const std::string& content,
                                      const std::string& key,
                                      const std::string& fallback) {
    const size_t value_pos = FindOptionalKey(content, key);
    return value_pos == std::string::npos
        ? fallback : ParseStringValue(content, value_pos, key);
}

static int ParseOptionalIntField(const std::string& content, const std::string& key,
                          int fallback) {
    return FindOptionalKey(content, key) == std::string::npos
        ? fallback : ParseRequiredInt(content, key);
}

static size_t FindCollectionEnd(const std::string& content, size_t start,
                          char open, char close, const std::string& field) {
    size_t depth = 0;
    bool in_string = false;
    for (size_t pos = start; pos < content.size(); ++pos) {
        const char c = content[pos];
        if (in_string && c == '\\') {
            ++pos;
        } else if (c == '"') {
            in_string = !in_string;
        } else if (!in_string) {
            if (c == open) {
                ++depth;
            } else if (c == close && --depth == 0) {
                return pos;
            }
        }
    }
    throw std::runtime_error("JSON 容器未闭合，字段: " + field
        + "，起始符号: '" + open + "'，缺少闭合符号: '" + close
        + "'，未闭合层数: " + std::to_string(depth)
        + "，输入大小: " + std::to_string(content.size()) + " 字节"
        + InputContext(content, start));
}

static std::string ParseObjectForKey(const std::string& content, const std::string& key) {
    const size_t start = FindOptionalKey(content, key);
    if (start == std::string::npos) {
        return "";
    }
    if (content[start] != '{') {
        throw std::runtime_error("字段必须为 JSON 对象，字段: " + key
            + "，预期 '{'" + InputContext(content, start));
    }
    const size_t end = FindCollectionEnd(content, start, '{', '}', key);
    return content.substr(start, end - start + 1);
}

static TokenizerInfo ParseTokenizerJson(const std::string& content) {
    TokenizerInfo tokenizer;
    const std::string obj = ParseObjectForKey(content, "tokenizer");
    if (obj.empty()) {
        return tokenizer;
    }
    const std::string type = ParseOptionalStringField(obj, "type", "ascii");
    if (type == "ascii") {
        tokenizer.type = TokenizerType::Ascii;
    } else if (type == "json_vocab") {
        tokenizer.type = TokenizerType::JsonVocab;
    } else {
        throw std::runtime_error("不支持的分词器类型: " + type
            + "，字段: tokenizer.type，支持的值: ascii、json_vocab");
    }
    tokenizer.path = ParseOptionalStringField(obj, "path", tokenizer.path);
    tokenizer.bos_id = ParseOptionalIntField(obj, "bos_id", tokenizer.bos_id);
    tokenizer.eos_id = ParseOptionalIntField(obj, "eos_id", tokenizer.eos_id);
    tokenizer.unk_id = ParseOptionalIntField(obj, "unk_id", tokenizer.unk_id);
    if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
        throw std::runtime_error("分词器特殊 token ID 必须为非负数"
            "，bos_id: " + std::to_string(tokenizer.bos_id)
            + "，eos_id: " + std::to_string(tokenizer.eos_id)
            + "，unk_id: " + std::to_string(tokenizer.unk_id));
    }
    if (tokenizer.bos_id == tokenizer.eos_id ||
        tokenizer.bos_id == tokenizer.unk_id ||
        tokenizer.eos_id == tokenizer.unk_id) {
        throw std::runtime_error("分词器特殊 token ID 必须互不相同"
            "，bos_id: " + std::to_string(tokenizer.bos_id)
            + "，eos_id: " + std::to_string(tokenizer.eos_id)
            + "，unk_id: " + std::to_string(tokenizer.unk_id));
    }
    if (tokenizer.type == TokenizerType::JsonVocab && tokenizer.path.empty()) {
        throw std::runtime_error("json_vocab 分词器必须提供路径"
            "，字段: tokenizer.path，实际值为空或缺失，预期非空字符串");
    }
    return tokenizer;
}

static void CheckConfig(const ModelConfig& config) {
    if (config.vocab_size == 0 || config.dim == 0 || config.hidden_dim == 0 ||
        config.layers_num == 0 || config.heads_num == 0 ||
        config.kv_heads_num == 0 || config.max_seq_len == 0) {
        throw std::runtime_error("所有模型维度必须为正数"
            "，vocab_size: " + std::to_string(config.vocab_size)
            + "，dim: " + std::to_string(config.dim)
            + "，hidden_dim: " + std::to_string(config.hidden_dim)
            + "，n_layers: " + std::to_string(config.layers_num)
            + "，n_heads: " + std::to_string(config.heads_num)
            + "，n_kv_heads: " + std::to_string(config.kv_heads_num)
            + "，max_seq_len: " + std::to_string(config.max_seq_len));
    }
    if (config.dim % config.heads_num != 0) {
        throw std::runtime_error("dim 必须能被 n_heads 整除"
            "，dim: " + std::to_string(config.dim)
            + "，n_heads: " + std::to_string(config.heads_num)
            + "，余数: " + std::to_string(config.dim % config.heads_num));
    }
    if (config.heads_num % config.kv_heads_num != 0) {
        throw std::runtime_error("n_heads 必须能被 n_kv_heads 整除"
            "，n_heads: " + std::to_string(config.heads_num)
            + "，n_kv_heads: " + std::to_string(config.kv_heads_num)
            + "，余数: " + std::to_string(config.heads_num % config.kv_heads_num));
    }
    if (config.head_dim == 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("head_dim 必须为正偶数"
            "，head_dim: " + std::to_string(config.head_dim)
            + "，dim: " + std::to_string(config.dim)
            + "，n_heads: " + std::to_string(config.heads_num));
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        throw std::runtime_error("rope_theta 必须为正数且有限"
            "，实际值: " + FloatToString(config.rope_theta));
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
        throw std::runtime_error("rms_norm_eps 必须为正数且有限"
            "，实际值: " + FloatToString(config.rms_norm_eps));
    }
}

static ModelConfig ParseConfigJson(const std::string& content) {
    ModelConfig config;
    config.vocab_size = ParseRequiredDimension(content, "vocab_size");
    config.dim = ParseRequiredDimension(content, "dim");
    config.hidden_dim = ParseRequiredDimension(content, "hidden_dim");
    config.layers_num = ParseRequiredDimension(content, "n_layers");
    config.heads_num = ParseRequiredDimension(content, "n_heads");
    config.kv_heads_num = ParseRequiredDimension(content, "n_kv_heads");
    config.max_seq_len = ParseRequiredDimension(content, "max_seq_len");
    config.rope_theta = ParseRequiredFloat(content, "rope_theta");
    config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");
    config.head_dim = config.dim / config.heads_num;
    CheckConfig(config);
    return config;
}

static bool TryParseLayerTensorKind(const std::string& name, TensorKind& kind) {
    if (name == "attention_norm") kind = TensorKind::AttentionNorm;
    else if (name == "wq") kind = TensorKind::Wq;
    else if (name == "wk") kind = TensorKind::Wk;
    else if (name == "wv") kind = TensorKind::Wv;
    else if (name == "wo") kind = TensorKind::Wo;
    else if (name == "ffn_norm") kind = TensorKind::FfnNorm;
    else if (name == "w_gate") kind = TensorKind::WGate;
    else if (name == "w_up") kind = TensorKind::WUp;
    else if (name == "w_down") kind = TensorKind::WDown;
    else return false;
    return true;
}

static TensorId ParseTensorId(const std::string& name) {
    if (name == "token_embedding") return {TensorKind::TokenEmbedding, -1};
    if (name == "final_norm") return {TensorKind::FinalNorm, -1};
    if (name == "lm_head") return {TensorKind::LmHead, -1};
    if (name.compare(0, 7, "layers.") == 0) {
        const size_t dot = name.find('.', 7);
        if (dot != std::string::npos) {
            const std::string layer_text = name.substr(7, dot - 7);
            const int layer = ParseStrictIntToken(layer_text, "张量 '" + name + "' 的层号");
            if (layer >= 0 && layer_text == std::to_string(layer)) {
                const std::string kind_name = name.substr(dot + 1);
                TensorKind kind;
                if (TryParseLayerTensorKind(kind_name, kind)) {
                    return {kind, layer};
                }
            }
        }
    }
    throw std::runtime_error("不支持的张量名称: " + name
        + "，预期 token_embedding、final_norm、lm_head 或 layers.<非负整数>."
        "<attention_norm|wq|wk|wv|wo|ffn_norm|w_gate|w_up|w_down>");
}

static std::string TensorName(const TensorId& id) {
    const char* kind_name = nullptr;
    switch (id.kind) {
        case TensorKind::TokenEmbedding: return "token_embedding";
        case TensorKind::FinalNorm: return "final_norm";
        case TensorKind::LmHead: return "lm_head";
        case TensorKind::AttentionNorm: kind_name = "attention_norm"; break;
        case TensorKind::Wq: kind_name = "wq"; break;
        case TensorKind::Wk: kind_name = "wk"; break;
        case TensorKind::Wv: kind_name = "wv"; break;
        case TensorKind::Wo: kind_name = "wo"; break;
        case TensorKind::FfnNorm: kind_name = "ffn_norm"; break;
        case TensorKind::WGate: kind_name = "w_gate"; break;
        case TensorKind::WUp: kind_name = "w_up"; break;
        case TensorKind::WDown: kind_name = "w_down"; break;
        default: break;
    }
    if (kind_name != nullptr) {
        return "layers." + std::to_string(id.layer) + "." + kind_name;
    }
    throw std::runtime_error("不支持的张量类型"
        "，类型枚举值: " + std::to_string(static_cast<int>(id.kind))
        + "，层号: " + std::to_string(id.layer));
}

static std::string ShapeToString(const std::vector<size_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    return result + "]";
}

static size_t CheckTensorByteSize(const TensorInfo& info) {
    size_t num_elements = 1;
    const std::string name = TensorName(info.id);
    for (size_t axis = 0; axis < info.shape.size(); ++axis) {
        const size_t dim = info.shape[axis];
        if (dim == 0) {
            throw std::runtime_error("张量形状无效，张量: " + name
                + "，轴号: " + std::to_string(axis)
                + "，实际维度: 0，预期正数维度"
                + "，形状: " + ShapeToString(info.shape));
        }
        if (num_elements > std::numeric_limits<size_t>::max() / dim) {
            throw std::runtime_error("张量元素总数溢出，张量: " + name
                + "，轴号: " + std::to_string(axis)
                + "，已累计元素数: " + std::to_string(num_elements)
                + "，当前维度: " + std::to_string(dim)
                + "，size_t 上限: " + std::to_string(std::numeric_limits<size_t>::max())
                + "，形状: " + ShapeToString(info.shape));
        }
        num_elements *= dim;
    }
    if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("张量字节大小溢出，张量: " + name
            + "，元素数: " + std::to_string(num_elements)
            + "，每个 float32 元素的字节数: " + std::to_string(sizeof(float))
            + "，size_t 上限: " + std::to_string(std::numeric_limits<size_t>::max())
            + "，形状: " + ShapeToString(info.shape));
    }
    return num_elements * sizeof(float);
}

static std::vector<std::pair<TensorId, std::vector<size_t>>>
ExpectedTensorShapes(const ModelConfig& config) {
    std::vector<std::pair<TensorId, std::vector<size_t>>> expected = {
        {{TensorKind::TokenEmbedding, -1}, {config.vocab_size, config.dim}},
        {{TensorKind::FinalNorm, -1}, {config.dim}},
        {{TensorKind::LmHead, -1}, {config.vocab_size, config.dim}},
    };
    for (size_t layer = 0; layer < config.layers_num; ++layer) {
        const int index = static_cast<int>(layer);
        const size_t q_dim = config.heads_num * config.head_dim;
        const size_t kv_dim = config.kv_heads_num * config.head_dim;
        expected.push_back({{TensorKind::AttentionNorm, index}, {config.dim}});
        expected.push_back({{TensorKind::Wq, index}, {q_dim, config.dim}});
        expected.push_back({{TensorKind::Wk, index}, {kv_dim, config.dim}});
        expected.push_back({{TensorKind::Wv, index}, {kv_dim, config.dim}});
        expected.push_back({{TensorKind::Wo, index}, {config.dim, q_dim}});
        expected.push_back({{TensorKind::FfnNorm, index}, {config.dim}});
        expected.push_back({{TensorKind::WGate, index}, {config.hidden_dim, config.dim}});
        expected.push_back({{TensorKind::WUp, index}, {config.hidden_dim, config.dim}});
        expected.push_back({{TensorKind::WDown, index}, {config.dim, config.hidden_dim}});
    }
    return expected;
}

static std::vector<TensorInfo> ParseTensorMetadata(const std::string& content) {
    const size_t start = FindConfigKey(content, "tensors");
    if (content[start] != '[') {
        throw std::runtime_error("tensors 必须为数组"
            "，预期 '['" + InputContext(content, start));
    }
    const size_t end = FindCollectionEnd(content, start, '[', ']', "tensors");
    std::vector<TensorInfo> result;
    size_t pos = SkipWhitespace(content, start + 1);
    while (pos < end) {
        if (!result.empty()) {
            if (content[pos] != ',') {
                throw std::runtime_error("张量条目之间缺少逗号 ','"
                    "，下一个张量序号: " + std::to_string(result.size())
                    + InputContext(content, pos));
            }
            pos = SkipWhitespace(content, pos + 1);
        }
        if (pos >= end || content[pos] != '{') {
            throw std::runtime_error("张量条目必须为 JSON 对象"
                "，张量序号: " + std::to_string(result.size())
                + "，预期 '{'" + InputContext(content, pos));
        }
        const size_t obj_end = FindCollectionEnd(content, pos, '{', '}',
            "tensors[" + std::to_string(result.size()) + "]");
        if (obj_end >= end) {
            throw std::runtime_error("张量对象未在 tensors 数组内闭合"
                "，张量序号: " + std::to_string(result.size())
                + "，对象起始字节位置: " + std::to_string(pos)
                + "，对象闭合字节位置: " + std::to_string(obj_end)
                + "，tensors 数组闭合字节位置: " + std::to_string(end));
        }
        const std::string obj = content.substr(pos, obj_end - pos + 1);
        std::string name;
        try {
            name = ParseRequiredStringField(obj, "name");
            TensorInfo info;
            info.id = ParseTensorId(name);
            const std::string dtype = ParseRequiredStringField(obj, "dtype");
            if (dtype != "float32") {
                throw std::runtime_error("不支持的 dtype '" + dtype + "'，张量: " + name
                    + "，支持的 dtype: float32");
            }
            info.dataType = DataType::Float32;

            const size_t shape_start = FindConfigKey(obj, "shape");
            if (obj[shape_start] != '[') {
                throw std::runtime_error("张量 shape 必须为数组，张量: " + name
                    + "，预期 '['" + InputContext(obj, shape_start));
            }
            const size_t shape_end = FindCollectionEnd(obj, shape_start, '[', ']', name + ".shape");
            const std::string shape_text = TrimWs(
                obj.substr(shape_start + 1, shape_end - shape_start - 1));
            if (!shape_text.empty() && shape_text.back() == ',') {
                throw std::runtime_error("张量形状无效，张量: " + name
                    + "，']' 前不能有多余的逗号"
                    + InputContext(obj, shape_start));
            }
            std::stringstream shape_stream(shape_text);
            std::string token;
            while (std::getline(shape_stream, token, ',')) {
                const std::string field = "张量 " + name + " 的 shape"
                    + "，轴号: " + std::to_string(info.shape.size());
                const int dim = ParseStrictIntToken(token, field);
                if (dim <= 0) {
                    throw std::runtime_error("张量形状无效，张量: " + name
                        + "，轴号: " + std::to_string(info.shape.size())
                        + "，实际维度: " + std::to_string(dim)
                        + "，预期正数维度"
                        + InputContext(obj, shape_start));
                }
                info.shape.push_back(static_cast<size_t>(dim));
            }
            info.offset = ParseStrictSizeValue(obj, FindConfigKey(obj, "offset"), "offset");
            info.byte_size = ParseStrictSizeValue(
                obj, FindConfigKey(obj, "byte_size"), "byte_size");
            result.push_back(info);
        } catch (const std::runtime_error& error) {
            throw std::runtime_error("tensors[" + std::to_string(result.size()) + "]"
                + (name.empty() ? "" : "（名称: '" + name + "'）")
                + "，对象在描述文件中的起始字节位置: " + std::to_string(pos)
                + "，原因: " + error.what());
        }
        pos = SkipWhitespace(content, obj_end + 1);
    }
    return result;
}

static void CheckDescriptor(const ModelDescriptor& descriptor) {
    if (descriptor.tensors.empty()) {
        throw std::runtime_error("模型描述中没有张量"
            "，字段: tensors，实际数量: 0，预期 3 个全局张量及每层 9 个张量"
            "，n_layers: " + std::to_string(descriptor.config.layers_num));
    }
    std::map<TensorId, const TensorInfo*> tensor_map;
    std::vector<const TensorInfo*> by_offset;
    by_offset.reserve(descriptor.tensors.size());
    for (const auto& info : descriptor.tensors) {
        const std::string name = TensorName(info.id);
        if (info.id.layer >= 0 &&
            static_cast<size_t>(info.id.layer) >= descriptor.config.layers_num) {
            throw std::runtime_error("张量层号越界，张量: " + name
                + "，层号: " + std::to_string(info.id.layer)
                + "，有效范围: [0, "
                + std::to_string(descriptor.config.layers_num) + ")");
        }
        if (info.dataType != DataType::Float32) {
            throw std::runtime_error("不支持的 dtype，张量: " + name
                + "，dtype 枚举值: " + std::to_string(static_cast<int>(info.dataType))
                + "，预期 Float32（枚举值: "
                + std::to_string(static_cast<int>(DataType::Float32)) + "）");
        }
        if (info.shape.empty()) {
            throw std::runtime_error("张量形状不能为空，张量: " + name
                + "，实际形状: []，预期至少包含一个正数维度");
        }
        const size_t expected_bytes = CheckTensorByteSize(info);
        if (expected_bytes != info.byte_size) {
            throw std::runtime_error("张量 byte_size 不匹配，张量: " + name +
                "，预期字节数: " + std::to_string(expected_bytes) +
                "，实际 byte_size: " + std::to_string(info.byte_size)
                + "，形状: " + ShapeToString(info.shape)
                + "，dtype: float32，每个元素的字节数: " + std::to_string(sizeof(float)));
        }
        if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
            throw std::runtime_error("张量偏移量溢出，张量: " + name
                + "，offset: " + std::to_string(info.offset)
                + "，byte_size: " + std::to_string(info.byte_size)
                + "，当前 byte_size 允许的最大 offset: "
                + std::to_string(std::numeric_limits<size_t>::max() - info.byte_size)
                + "，size_t 上限: " + std::to_string(std::numeric_limits<size_t>::max()));
        }
        const auto inserted = tensor_map.emplace(info.id, &info);
        if (!inserted.second) {
            throw std::runtime_error("张量名称重复: " + name
                + "，首次出现的 offset: " + std::to_string(inserted.first->second->offset)
                + "，重复条目的 offset: " + std::to_string(info.offset)
                + "，每个张量名称必须恰好对应一个条目");
        }
        by_offset.push_back(&info);
    }
    std::sort(by_offset.begin(), by_offset.end(),
        [](const TensorInfo* a, const TensorInfo* b) { return a->offset < b->offset; });
    for (size_t i = 1; i < by_offset.size(); ++i) {
        const TensorInfo& previous = *by_offset[i - 1];
        const TensorInfo& current = *by_offset[i];
        if (current.offset < previous.offset + previous.byte_size) {
            throw std::runtime_error("张量字节区间重叠: " + TensorName(previous.id) +
                " [" + std::to_string(previous.offset) + ", "
                + std::to_string(previous.offset + previous.byte_size) + ") 与 "
                + TensorName(current.id) + " [" + std::to_string(current.offset) + ", "
                + std::to_string(current.offset + current.byte_size) + ")"
                + "，各张量的字节区间必须互不重叠");
        }
    }

    if (descriptor.tensors.size() < 3 ||
        descriptor.config.layers_num > (descriptor.tensors.size() - 3) / 9) {
        throw std::runtime_error("缺少必需张量，无法满足配置层数"
            "，n_layers: " + std::to_string(descriptor.config.layers_num)
            + "，预期张量数: "
            + std::to_string(3ULL + 9ULL * descriptor.config.layers_num)
            + "（3 个全局张量 + 每层 9 个），实际张量数: "
            + std::to_string(descriptor.tensors.size()));
    }
    for (const auto& expected : ExpectedTensorShapes(descriptor.config)) {
        const auto it = tensor_map.find(expected.first);
        const std::string name = TensorName(expected.first);
        if (it == tensor_map.end()) {
            throw std::runtime_error("缺少必需张量: " + name
                + "，预期形状: " + ShapeToString(expected.second)
                + "，n_layers: " + std::to_string(descriptor.config.layers_num)
                + "，实际张量数: " + std::to_string(descriptor.tensors.size()));
        }
        if (it->second->shape != expected.second) {
            throw std::runtime_error("张量形状不匹配，张量: " + name +
                "，预期形状: " + ShapeToString(expected.second) +
                "，实际形状: " + ShapeToString(it->second->shape)
                + "，dim: " + std::to_string(descriptor.config.dim)
                + "，hidden_dim: " + std::to_string(descriptor.config.hidden_dim)
                + "，vocab_size: " + std::to_string(descriptor.config.vocab_size)
                + "，n_heads: " + std::to_string(descriptor.config.heads_num)
                + "，n_kv_heads: " + std::to_string(descriptor.config.kv_heads_num)
                + "，head_dim: " + std::to_string(descriptor.config.head_dim));
        }
    }
}

ModelDescriptor ParseDescriptor(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("[Loader] 调用 ParseDescriptor 时无法打开配置文件，路径: '"
            + config_path + "'，预期可读取的模型描述文件");
    }
    std::stringstream buffer;
    buffer << config_file.rdbuf();
    if (config_file.bad()) {
        throw std::runtime_error("[Loader] 调用 ParseDescriptor 时读取配置文件失败，路径: '"
            + config_path + "'，文件流已设置 badbit，已读取字节数: "
            + std::to_string(buffer.str().size()));
    }
    const std::string content = buffer.str();
    try {
        ModelDescriptor descriptor;
        descriptor.config = ParseConfigJson(content);
        descriptor.tokenizer = ParseTokenizerJson(content);
        descriptor.tensors = ParseTensorMetadata(content);
        CheckDescriptor(descriptor);
        return descriptor;
    } catch (const std::runtime_error& error) {
        throw std::runtime_error("[Loader] 调用 ParseDescriptor 时解析模型描述失败，路径: '"
            + config_path + "'，原因: " + error.what());
    }
}

static Tensor LoadTensorById(std::ifstream& wf,
                             const std::map<TensorId, TensorInfo>& tensors_map,
                             const TensorId& id) {
    const auto it = tensors_map.find(id);
    if (it == tensors_map.end()) {
        throw std::runtime_error("缺少必需张量: " + TensorName(id));
    }

    const TensorInfo& info = it->second;

    if (info.offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("tensor offset exceeds stream limit for " + TensorName(id));
    }
    if (info.byte_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor byte size exceeds stream limit for " + TensorName(id));
    }

    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    if (!wf.good()) {
        throw std::runtime_error("定位权重失败，张量: " + TensorName(id)
            + "，offset: " + std::to_string(info.offset));
    }
    Tensor t(info.shape, 0.0f);
    wf.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(info.byte_size));

    if (!wf || static_cast<size_t>(wf.gcount()) != info.byte_size) {
        throw std::runtime_error("读取权重失败，张量: " + TensorName(id)
            + "，offset: " + std::to_string(info.offset)
            + "，预期字节数: " + std::to_string(info.byte_size)
            + "，实际读取字节数: " + std::to_string(wf.gcount()));
    }
    return t;
}

Model LoadModel(const std::string& config_path, const std::string& weights_path) {
    ModelDescriptor descriptor;

    try {
        descriptor = ParseDescriptor(config_path);
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }

    std::cout << "Loaded config: vocab=" << descriptor.config.vocab_size
              << " dim=" << descriptor.config.dim
              << " layers=" << descriptor.config.layers_num
              << " heads=" << descriptor.config.heads_num
              << " kv_heads=" << descriptor.config.kv_heads_num
              << " head_dim=" << descriptor.config.head_dim
              << " hidden_dim=" << descriptor.config.hidden_dim << std::endl;
    std::map<TensorId, TensorInfo> tensors_map;
    for (const auto& info : descriptor.tensors){
        tensors_map[info.id] = info;
    }

    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf.is_open()){
        return LoadFailure("fail to open weights: " + weights_path);
    }

    Model model;
    model.config = descriptor.config;
    try {
        model.token_embedding = LoadTensorById(
            wf, tensors_map, {TensorKind::TokenEmbedding, -1});
        model.layers.resize(model.config.layers_num);
        for (size_t i = 0; i < model.layers.size(); ++i) {
            LayerWeights& layer = model.layers[i];
            const int index = static_cast<int>(i);
            layer.attention_norm = LoadTensorById(
                wf, tensors_map, {TensorKind::AttentionNorm, index});
            layer.wq = LoadTensorById(wf, tensors_map, {TensorKind::Wq, index});
            layer.wk = LoadTensorById(wf, tensors_map, {TensorKind::Wk, index});
            layer.wv = LoadTensorById(wf, tensors_map, {TensorKind::Wv, index});
            layer.wo = LoadTensorById(wf, tensors_map, {TensorKind::Wo, index});
            layer.ffn_norm = LoadTensorById(
                wf, tensors_map, {TensorKind::FfnNorm, index});
            layer.w_gate = LoadTensorById(wf, tensors_map, {TensorKind::WGate, index});
            layer.w_up = LoadTensorById(wf, tensors_map, {TensorKind::WUp, index});
            layer.w_down = LoadTensorById(wf, tensors_map, {TensorKind::WDown, index});
        }
        model.final_norm = LoadTensorById(wf, tensors_map, {TensorKind::FinalNorm, -1});
        model.lm_head = LoadTensorById(wf, tensors_map, {TensorKind::LmHead, -1});
    } catch (const std::exception& e) {
        return LoadFailure(e.what());
    }

    wf.seekg(0, std::ios::end);
    std::streamoff file_size = wf.tellg();
    if (file_size < 0){
        return LoadFailure("failed to determine weights file size");
    }

    size_t expected_end = 0;
    for (const auto& info : descriptor.tensors){
        size_t tensor_end = info.offset + info.byte_size;
        if (tensor_end > expected_end){
            expected_end = tensor_end;
        }
    }

    if (static_cast<size_t>(file_size) != expected_end) {
        if (static_cast<size_t>(file_size) > expected_end) {
            return LoadFailure("weights file has trailing bytes: file size=" + std::to_string(file_size) +
                               ", expected=" + std::to_string(expected_end));
        } else {
            return LoadFailure("weights file is too short: file size=" + std::to_string(file_size) +
                               ", expected=" + std::to_string(expected_end));
        }
    }

    model.loaded = true;
    std::cout << "Model loaded successfully from " << weights_path << std::endl;
    return model;
}
