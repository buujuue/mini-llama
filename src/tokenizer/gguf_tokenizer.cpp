// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "gguf_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gguf.h"

namespace {


static std::string CodepointToUtf8(char32_t cp) {
  std::string result;
  if (cp <= 0x7F) {
    result += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    result += static_cast<char>(0xC0 | (cp >> 6));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (cp >> 12));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (cp >> 18));
    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return result;
}

static std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
  if (pos >= s.size()) {
    return "";
  }
  unsigned char first = s[pos];
  size_t len = 1;
  if ((first & 0x80) == 0) {
    len = 1;
  } else if ((first & 0xE0) == 0xC0) {
    len = 2;
  } else if ((first & 0xF0) == 0xE0) {
    len = 3;
  } else if ((first & 0xF8) == 0xF0) {
    len = 4;
  } else {
    ++pos;
    return "";
  }
  if (pos + len > s.size()) {
    ++pos;
    return "";
  }
  std::string result = s.substr(pos, len);
  pos += len;
  return result;
}


static std::vector<std::string> BuildBytesToUnicode() {
  std::vector<std::string> map(256);

  std::vector<int> bs;
  for (int c = '!'; c <= '~'; ++c) {
    bs.push_back(c);
  }
  for (int c = 0xA1; c <= 0xAC; ++c) {
    bs.push_back(c);
  }
  for (int c = 0xAE; c <= 0xFF; ++c) {
    bs.push_back(c);
  }

  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }

  for (size_t i = 0; i < bs.size(); ++i) {
    map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
  }
  return map;
}

}  // namespace

void GgufTokenizer::BuildByteMappings() {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[b]] = static_cast<uint8_t>(b);
  }
}

bool GgufTokenizer::Load(const std::string& gguf_path) {
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;

  b2u_.clear();
  u2b_.clear();

  
  std::string model;
  if (GgufGetMetadataString(gguf_path, "tokenizer.ggml.model", model) && model != "gpt2") {
    std::cerr << "GGUF tokenizer 不支持 tokenizer.ggml.model=" << model << std::endl;
    return false;
  }

 
  std::vector<std::string> tokens;
  if (!GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.tokens", tokens) ||
      tokens.empty() || tokens.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::cerr << "GGUF 词表缺失或无效：" << gguf_path << std::endl;
    return false;
  }

  std::vector<int32_t> token_types;
  bool has_token_types = GgufGetMetadataIntArray(
      gguf_path, "tokenizer.ggml.token_type", token_types);


  std::unordered_map<std::string, int> vocab;
  vocab.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i].empty() || !vocab.emplace(tokens[i], static_cast<int>(i)).second) {
      std::cerr << "GGUF 包含空 token 或重复词表项，ID=" << i << std::endl;
      return false;
    }
  }


  std::vector<std::string> merges;
  GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.merges", merges);
  auto read_id = [&](const std::string& key, int& id) {
    int64_t value = -1;
    if (!GgufGetMetadataInt(gguf_path, key, value)) {
      return true;
    }
    if (value < -1 || value >= static_cast<int64_t>(tokens.size())) {
      std::cerr << "GGUF 特殊 token ID 越界：" << key << "=" << value << std::endl;
      return false;
    }
    id = static_cast<int>(value);
    return true;
  };
  int bos = -1, eos = -1, unk = -1;
  if (!read_id("tokenizer.ggml.bos_token_id", bos) ||
      !read_id("tokenizer.ggml.eos_token_id", eos) ||
      !read_id("tokenizer.ggml.unknown_token_id", unk)) {
    return false;
  }

  vocab_ = std::move(vocab);
  id_to_token_ = std::move(tokens);
  bos_id_ = bos;
  eos_id_ = eos;
  unk_id_ = unk;
  BuildByteMappings();


  if (has_token_types && token_types.size() == id_to_token_.size()) {
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
      if (token_types[i] == 3) {
        special_tokens_.push_back({id_to_token_[i], static_cast<int>(i)});
      }
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });

  for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
    const std::string& m = merges[rank];
    size_t space_pos = m.find(' ');
    if (space_pos == std::string::npos || space_pos == 0 || space_pos + 1 == m.size()) {
      continue;
    }
    std::string left = m.substr(0, space_pos);
    std::string right = m.substr(space_pos + 1);
    merge_ranks_[{left, right}] = rank;
  }

  std::cout << "GGUF tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size()
            << " special=" << special_tokens_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
  return true;
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  if (text.empty() || b2u_.empty()) {
    return {};
  }

  std::vector<int> result;
  size_t pos = 0;

  while (pos < text.size()) {

    bool matched = false;
    for (const auto& st : special_tokens_) {
      const std::string& token_str = st.first;
      if (pos + token_str.size() <= text.size() &&
          std::memcmp(text.data() + pos, token_str.data(), token_str.size()) ==
              0) {
        result.push_back(st.second);
        pos += token_str.size();
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    // Find the next special token or end of string
    size_t end = text.size();
    for (const auto& st : special_tokens_) {
      size_t p = text.find(st.first, pos);
      if (p != std::string::npos && p < end) {
        end = p;
      }
    }

    // Encode the plain text segment [pos, end) with BPE
    std::string segment = text.substr(pos, end - pos);
    pos = end;

    // 1. Convert text to initial word pieces (byte -> unicode mapping)
    std::vector<std::string> word;
    for (unsigned char c : segment) {
      if (c >= b2u_.size()) {
        std::cerr << "ERROR: byte " << static_cast<int>(c) << " >= b2u_ size "
                  << b2u_.size() << "\n";
        return {};
      }
      word.push_back(b2u_[c]);
    }

    // 2. Apply BPE merges
    while (word.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      size_t best_idx = word.size();  // invalid

      for (size_t i = 0; i + 1 < word.size(); ++i) {
        auto it = merge_ranks_.find({word[i], word[i + 1]});
        if (it != merge_ranks_.end() && it->second < best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }

      if (best_idx >= word.size()) {
        break;  // no more merges possible
      }

      word[best_idx] += word[best_idx + 1];
      word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
    }

    // 3. Map to token IDs
    for (const auto& w : word) {
      auto it = vocab_.find(w);
      if (it != vocab_.end()) {
        result.push_back(it->second);
      } else {
        result.push_back(unk_id_);
      }
    }
  }

  return result;
}

std::string GgufTokenizer::DecodeToken(int token) const {
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[token];
  }
  return "";
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  // 1. Concatenate token strings
  std::string text;
  for (int id : tokens) {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
      text += id_to_token_[id];
    }
  }

  // 2. Convert unicode chars back to bytes
  std::string result;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t start = pos;
    std::string cp = NextUtf8Codepoint(text, pos);
    if (cp.empty()) {
      if (pos == start) {
        ++pos;
      }
      continue;
    }
    auto it = u2b_.find(cp);
    if (it != u2b_.end()) {
      result += static_cast<char>(it->second);
    } else {
      result += cp;
    }
  }

  return result;
}


std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
  auto tok = std::make_unique<GgufTokenizer>();
  if (tok->Load(gguf_path)) {
    return tok;
  }
  return nullptr;
}

