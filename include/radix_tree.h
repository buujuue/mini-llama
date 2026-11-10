
#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>


class RadixTree {
public:
    RadixTree();
    ~RadixTree();

    RadixTree(const RadixTree&) = delete;
    RadixTree& operator=(const RadixTree&) = delete;
 
    RadixTree(RadixTree&& other) noexcept;
    RadixTree& operator=(RadixTree&& other) noexcept;

    void Clear();
    bool empty() const;
   
    std::size_t size() const;

    void Insert(const std::vector<int>& tokens);
    
    std::size_t LongestPrefix(const std::vector<int>& tokens) const;

private:
    struct Node {
        std::vector<int> key;
        bool terminal = false;
        std::map<int, std::unique_ptr<Node>> children;
    };

    static std::size_t CommonPrefixLength(const std::vector<int>& a,
                                         std::size_t a_offset,
                                         const std::vector<int>& b);
    void InsertInto(Node* parent, std::vector<int> suffix);
        
    Node root_;
    std::size_t terminal_count_ = 0;
};
