#include <cstddef>
#include <vector>

class batch{
public:
    batch() = delete;
    batch(int token, size_t pos = 0);
    batch(const std::vector<int>& tokens, size_t start_pos);
    size_t size(){return tokens_.size();}
private:
    std::vector<int> tokens_;
    std::vector<size_t> positions_;
};