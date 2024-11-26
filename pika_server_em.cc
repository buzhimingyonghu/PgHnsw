#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Product Quantization (PQ) + HNSW 实现
class PQHNSW {
 private:
  static constexpr int ORIG_DIM = 128;                          // 原始维度
  static constexpr int REDUCED_DIM = 8;                         // 压缩后维度
  static constexpr int SUBVECTOR_DIM = ORIG_DIM / REDUCED_DIM;  // 每个子向量的维度

  struct Node {
    int id;                                               // 节点 ID
    std::vector<uint8_t> compressed;                      // 压缩后的向量（Codebook 索引）
    std::unordered_map<int, std::vector<int>> neighbors;  // 邻居，按层次存储
  };

  std::vector<Node> nodes_;                                // 所有节点
  std::vector<std::vector<std::vector<float>>> codebook_;  // Codebook
  int max_neighbors_;                                      // 每层的最大邻居数
  int max_level_;                                          // 最大层数
  float ef_;                                               // 搜索时扩展的邻居数量

  // 随机初始化 Codebook
  void InitializeCodebook() {
    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    codebook_.resize(REDUCED_DIM);
    for (int i = 0; i < REDUCED_DIM; ++i) {
      codebook_[i].resize(256, std::vector<float>(SUBVECTOR_DIM));
      for (auto& centroid : codebook_[i]) {
        for (float& val : centroid) {
          val = distribution(engine);
        }
      }
    }
  }

  // 计算两个向量的欧几里得距离
  static float EuclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float distance = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
      distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(distance);
  }

  // 量化向量
  std::vector<uint8_t> QuantizeVector(const std::vector<float>& vec) {
    std::vector<uint8_t> compressed(REDUCED_DIM);
    for (int i = 0; i < REDUCED_DIM; ++i) {
      float min_dist = std::numeric_limits<float>::max();
      uint8_t best_index = 0;

      for (uint8_t j = 0; j < 256; ++j) {
        float dist = EuclideanDistance({vec.begin() + i * SUBVECTOR_DIM, vec.begin() + (i + 1) * SUBVECTOR_DIM},
                                       codebook_[i][j]);
        if (dist < min_dist) {
          min_dist = dist;
          best_index = j;
        }
      }
      compressed[i] = best_index;
    }
    return compressed;
  }

  // 解码向量
  std::vector<float> DecodeVector(const std::vector<uint8_t>& compressed) {
    std::vector<float> reconstructed(ORIG_DIM);
    for (int i = 0; i < REDUCED_DIM; ++i) {
      const auto& centroid = codebook_[i][compressed[i]];
      std::copy(centroid.begin(), centroid.end(), reconstructed.begin() + i * SUBVECTOR_DIM);
    }
    return reconstructed;
  }

  // 搜索候选集
  std::vector<int> SearchLayer(const std::vector<float>& query, int enter_point, int level) {
    std::priority_queue<std::pair<float, int>> candidate_queue;
    std::unordered_set<int> visited;

    candidate_queue.emplace(-EuclideanDistance(query, DecodeVector(nodes_[enter_point].compressed)), enter_point);
    visited.insert(enter_point);

    std::vector<int> result;
    while (!candidate_queue.empty() && result.size() < ef_) {
      auto [neg_dist, current_id] = candidate_queue.top();
      candidate_queue.pop();

      result.push_back(current_id);
      for (int neighbor : nodes_[current_id].neighbors[level]) {
        if (visited.count(neighbor)) continue;
        visited.insert(neighbor);
        float dist = EuclideanDistance(query, DecodeVector(nodes_[neighbor].compressed));
        candidate_queue.emplace(-dist, neighbor);
      }
    }

    return result;
  }

  // 随机生成层数
  int GenerateRandomLevel() {
    static std::default_random_engine engine(std::random_device{}());
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    int level = 0;
    while (distribution(engine) < 0.5f && level < max_level_) {
      ++level;
    }
    return level;
  }

 public:
  PQHNSW(int max_neighbors = 16, int max_level = 5, float ef = 10)
      : max_neighbors_(max_neighbors), max_level_(max_level), ef_(ef) {
    InitializeCodebook();
  }

  // 添加一个节点
  void AddNode(int id, const std::vector<float>& data) {
    Node new_node{id, QuantizeVector(data), {}};
    int level = GenerateRandomLevel();
    for (int i = 0; i <= level; ++i) {
      new_node.neighbors[i] = {};
    }
    if (nodes_.empty()) {
      nodes_.push_back(new_node);
      return;
    }

    int enter_point = 0;  // 初始进入点
    for (int l = max_level_; l >= 0; --l) {
      enter_point = SearchLayer(data, enter_point, l).front();
    }

    nodes_.push_back(new_node);
    int new_id = nodes_.size() - 1;

    for (int l = 0; l <= level; ++l) {
      auto neighbors = SearchLayer(data, enter_point, l);
      neighbors.resize(std::min(neighbors.size(), static_cast<size_t>(max_neighbors_)));

      for (int neighbor : neighbors) {
        nodes_[neighbor].neighbors[l].push_back(new_id);
        nodes_[new_id].neighbors[l].push_back(neighbor);
      }
    }
  }

  // 搜索最接近的向量
  int Search(const std::vector<float>& query) {
    int enter_point = 0;
    for (int l = max_level_; l >= 0; --l) {
      auto candidates = SearchLayer(query, enter_point, l);
      if (!candidates.empty()) {
        enter_point = candidates.front();
      }
    }

    auto final_candidates = SearchLayer(query, enter_point, 0);
    return final_candidates.front();
  }
};

// 测试 PQ + HNSW
int main() {
  PQHNSW hnsw;

  // 添加节点
  std::vector<float> vec1(128, 1.0f);
  std::vector<float> vec2(128, 2.0f);
  std::vector<float> vec3(128, 3.0f);

  hnsw.AddNode(0, vec1);
  hnsw.AddNode(1, vec2);
  hnsw.AddNode(2, vec3);

  // 搜索最接近的向量
  std::vector<float> query(128, 1.5f);
  int closest = hnsw.Search(query);

  std::cout << "Closest node to query is: " << closest << std::endl;
  return 0;
}