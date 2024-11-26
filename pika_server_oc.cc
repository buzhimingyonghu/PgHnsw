
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eisther express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "../../default_allocator.h"
#include "../../simd/simd.h"
#include "../../utils.h"
#include "block_manager.h"
#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {
// 定义类型别名，tableint 表示无符号整型（通常用于表示表的索引）
using tableint = unsigned int;

// 定义 linklistsizeint 表示链表大小类型（无符号整型）
using linklistsizeint = unsigned int;

// 定义 reverselinklist 类型，表示一个无序集合，存储 uint32_t 类型的元素
using reverselinklist = vsag::UnorderedSet<uint32_t>;

// 比较函数对象，用于根据 pair 的第一个元素（float 类型）进行排序
struct CompareByFirst {
  constexpr bool operator()(std::pair<float, tableint> const& a, std::pair<float, tableint> const& b) const noexcept {
    return a.first < b.first;  // 比较两个 pair 的第一个元素
  }
};

// 定义最大堆类型 MaxHeap，使用优先队列实现，存储 pair<float, tableint>
// 使用 CompareByFirst 来比较堆中的元素
using MaxHeap =
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>;

// 定义常量，表示误差阈值
const static float THRESHOLD_ERROR = 1e-6;

// 定义 HierarchicalNSW 类，继承自 AlgorithmInterface，用于层次化近邻搜索
class HierarchicalNSW : public AlgorithmInterface<float> {
 private:
  static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;  // 最大标签操作锁数量
  static const unsigned char DELETE_MARK = 0x01;            // 删除标记

  // 私有成员变量
  size_t max_elements_{0};                            // 最大元素数量
  mutable std::atomic<size_t> cur_element_count_{0};  // 当前元素数量，使用原子变量保证线程安全
  size_t size_data_per_element_{0};                   // 每个元素的数据大小
  size_t size_links_per_element_{0};                  // 每个元素的链表大小
  mutable std::atomic<size_t> num_deleted_{0};        // 删除元素的数量
  size_t M_{0};                                       // 每个元素的最大邻居数量
  size_t maxM_{0};                                    // 最大邻居数量
  size_t maxM0_{0};                                   // 初始邻居数量
  size_t ef_construction_{0};                         // 构建时的 ef 值
  size_t dim_{0};                                     // 特征维度

  double mult_{0.0}, revSize_{0.0};  // 一些用于计算的浮动常量
  int maxlevel_{0};                  // 最大层数

  std::shared_ptr<VisitedListPool> visited_list_pool_{nullptr};  // 访问列表池，用于管理访问记录

  // 元素标签操作锁，防止多个线程同时操作同一个标签的元素
  mutable vsag::Vector<std::mutex> label_op_locks_;

  std::mutex global_{};                                 // 全局锁，控制全局共享资源
  vsag::Vector<std::recursive_mutex> link_list_locks_;  // 链表的递归锁，保证对链表的线程安全操作

  tableint enterpoint_node_{0};  // 入口节点的 ID

  // 链接和数据偏移量
  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  bool normalize_ = false;  // 是否进行归一化处理
  float* molds_ = nullptr;  // 模板数据

  std::shared_ptr<BlockManager> data_level0_memory_{nullptr};  // 数据块管理器，管理 level0 数据
  char** link_lists_{nullptr};                                 // 链接列表数组
  int* element_levels_{nullptr};                               // 存储每个元素所在的层级

  bool use_reversed_edges_ = false;                                          // 是否使用反向边
  reverselinklist** reversed_level0_link_list_{nullptr};                     // 反向边的 level0 链接列表
  vsag::UnorderedMap<int, reverselinklist>** reversed_link_lists_{nullptr};  // 存储反向边的链接列表映射

  size_t data_size_{0};               // 数据的总大小
  size_t data_element_per_block_{0};  // 每个块的数据元素数量

  DISTFUNC fstdistfunc_;            // 距离函数类型
  void* dist_func_param_{nullptr};  // 距离函数的额外参数

  // 锁住 label_lookup_，避免多个线程同时操作
  mutable std::mutex label_lookup_lock_{};
  vsag::UnorderedMap<labeltype, tableint> label_lookup_;  // 标签查找表

  std::default_random_engine level_generator_;               // 随机数生成器，用于生成层级
  std::default_random_engine update_probability_generator_;  // 随机数生成器，用于更新概率

  vsag::Allocator* allocator_{nullptr};  // 内存分配器

  mutable std::atomic<long> metric_distance_computations_{0};  // 计算的距离次数
  mutable std::atomic<long> metric_hops_{0};                   // 跳数统计

  vsag::DistanceFunc ip_func_;  // 内积函数，用于计算相似度

  bool allow_replace_deleted_ = false;  // 是否允许在插入时替换已删除的元素

  // 删除元素的锁
  std::mutex deleted_elements_lock_{};
  vsag::UnorderedSet<tableint> deleted_elements_;  // 存储已删除元素的集合

 public:
  // 构造函数，初始化 HierarchicalNSW 对象
  HierarchicalNSW(SpaceInterface* s, size_t max_elements, vsag::Allocator* allocator, size_t M = 16,
                  size_t ef_construction = 200, bool use_reversed_edges = false, bool normalize = false,
                  size_t block_size_limit = 128 * 1024 * 1024, size_t random_seed = 100,
                  bool allow_replace_deleted = false)
      : allocator_(allocator),
        link_list_locks_(max_elements, allocator),              // 初始化链表锁
        label_op_locks_(MAX_LABEL_OPERATION_LOCKS, allocator),  // 初始化标签操作锁
        allow_replace_deleted_(allow_replace_deleted),
        use_reversed_edges_(use_reversed_edges),
        normalize_(normalize),
        label_lookup_(allocator),                      // 标签查找表
        deleted_elements_(allocator) {                 // 删除元素集合
    max_elements_ = max_elements;                      // 最大元素数
    num_deleted_ = 0;                                  // 已删除元素数
    data_size_ = s->get_data_size();                   // 获取数据大小
    fstdistfunc_ = s->get_dist_func();                 // 获取距离函数
    dist_func_param_ = s->get_dist_func_param();       // 获取距离函数的参数
    dim_ = data_size_ / sizeof(float);                 // 数据维度
    M_ = M;                                            // 每个节点的最大邻居数
    maxM_ = M_;                                        // 最大邻居数
    maxM0_ = M_ * 2;                                   // 初始最大邻居数
    ef_construction_ = std::max(ef_construction, M_);  // 构建时的 ef 值

    level_generator_.seed(random_seed);                   // 随机数生成器种子
    update_probability_generator_.seed(random_seed + 1);  // 更新概率随机数种子

    // 计算每个节点的链接大小和数据大小
    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;                 // 数据偏移量
    label_offset_ = size_links_level0_ + data_size_;  // 标签偏移量
    offsetLevel0_ = 0;                                // level0 偏移量

    // 初始化内存管理器
    data_level0_memory_ = std::make_shared<BlockManager>(size_data_per_element_, block_size_limit, allocator_);
    data_element_per_block_ = block_size_limit / size_data_per_element_;  // 每个块的数据元素数

    cur_element_count_ = 0;  // 当前元素数

    visited_list_pool_ = std::make_shared<VisitedListPool>(1, max_elements, allocator_);  // 访问列表池

    // 初始化特殊节点的处理
    enterpoint_node_ = -1;                                                         // 入口节点
    maxlevel_ = -1;                                                                // 最大层数
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);  // 每个节点的链接大小
    mult_ = 1 / log(1.0 * M_);                                                     // 用于计算跳跃概率的常数
    revSize_ = 1.0 / mult_;                                                        // 反转跳跃概率
  }

  // 析构函数，释放资源
  ~HierarchicalNSW() {
    // 释放链表内存
    if (link_lists_ != nullptr) {
      for (tableint i = 0; i < max_elements_; i++) {
        if (element_levels_[i] > 0 || link_lists_[i] != nullptr) allocator_->Deallocate(link_lists_[i]);
      }
    }

    // 释放反向边内存
    if (use_reversed_edges_) {
      for (tableint i = 0; i < max_elements_; i++) {
        auto& in_edges_level0 = *(reversed_level0_link_list_ + i);
        if (in_edges_level0) {
          delete in_edges_level0;
        }
        auto& in_edges = *(reversed_link_lists_ + i);
        if (in_edges) {
          delete in_edges;
        }
      }
    }
    reset();  // 重置状态
  }

  // 向量归一化处理
  void normalize_vector(const void*& data_point, std::shared_ptr<float[]>& normalize_data) const {
    if (normalize_) {
      // 计算归一化模长
      float query_mold = std::sqrt(ip_func_(data_point, data_point, dist_func_param_));
      normalize_data.reset(new float[dim_]);
      for (int i = 0; i < dim_; ++i) {
        normalize_data[i] = ((float*)data_point)[i] / query_mold;  // 归一化每个元素
      }
      data_point = normalize_data.get();  // 更新数据指针为归一化后的数据
    }
  }

  // 通过标签获取距离
  float getDistanceByLabel(labeltype label, const void* data_point) override {
    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);

    // 查找标签对应的内部 ID
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");  // 标签未找到，抛出异常
    }
    tableint internal_id = search->second;
    lock_table.unlock();

    // 归一化查询向量并计算距离
    std::shared_ptr<float[]> normalize_query;
    normalize_vector(data_point, normalize_query);
    float dist = fstdistfunc_(data_point, getDataByInternalId(internal_id), dist_func_param_);
    return dist;
  }

  // 检查标签是否有效
  bool isValidLabel(labeltype label) override {
    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
    bool is_valid = (label_lookup_.find(label) != label_lookup_.end());  // 查找标签是否存在
    lock_table.unlock();
    return is_valid;
  }

  // 比较函数对象，用于按第一个元素（float）排序
  struct CompareByFirst {
    constexpr bool operator()(std::pair<float, tableint> const& a, std::pair<float, tableint> const& b) const noexcept {
      return a.first < b.first;
    }
  };

  // 获取标签操作锁
  inline std::mutex& getLabelOpMutex(labeltype label) const {
    size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);  // 计算锁的索引
    return label_op_locks_[lock_id];
  }

  // 获取外部标签
  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype value;
    std::memcpy(&value, data_level0_memory_->GetElementPtr(internal_id, label_offset_), sizeof(labeltype));
    return value;
  }

  // 设置外部标签
  inline void setExternalLabel(tableint internal_id, labeltype label) const {
    *(labeltype*)(data_level0_memory_->GetElementPtr(internal_id, label_offset_)) = label;
  }

  // 获取外部标签指针
  inline labeltype* getExternalLabeLp(tableint internal_id) const {
    return (labeltype*)(data_level0_memory_->GetElementPtr(internal_id, label_offset_));
  }

  // 获取边集合
  inline reverselinklist& getEdges(tableint internal_id, int level = 0) {
    if (level != 0) {
      auto& edge_map_ptr = reversed_link_lists_[internal_id];
      if (edge_map_ptr == nullptr) {
        edge_map_ptr = new vsag::UnorderedMap<int, reverselinklist>(allocator_);
      }
      auto& edge_map = *edge_map_ptr;
      if (edge_map.find(level) == edge_map.end()) {
        edge_map.insert(std::make_pair(level, reverselinklist(allocator_)));  // 插入新级别的边
      }
      return edge_map.at(level);
    } else {
      auto& edge_ptr = reversed_level0_link_list_[internal_id];
      if (edge_ptr == nullptr) {
        edge_ptr = new reverselinklist(allocator_);
      }
      return *edge_ptr;
    }
  }

  // 更新连接
  void updateConnections(tableint internal_id, const vsag::Vector<tableint>& cand_neighbors, int level,
                         bool is_update) {
    linklistsizeint* ll_cur;
    if (level == 0)
      ll_cur = get_linklist0(internal_id);
    else
      ll_cur = get_linklist(internal_id, level);

    auto cur_size = getListCount(ll_cur);
    tableint* data = (tableint*)(ll_cur + 1);

    if (is_update && use_reversed_edges_) {
      for (int i = 0; i < cur_size; ++i) {
        auto id = data[i];
        auto& in_edges = getEdges(id, level);
        // 移除指向当前节点的边
        std::unique_lock<std::recursive_mutex> lock(link_list_locks_[i]);
        in_edges.erase(internal_id);
      }
    }

    setListCount(ll_cur, cand_neighbors.size());  // 更新链表大小
    for (size_t i = 0; i < cand_neighbors.size(); i++) {
      auto id = cand_neighbors[i];
      data[i] = cand_neighbors[i];
      if (not use_reversed_edges_) {
        continue;
      }
      std::unique_lock<std::recursive_mutex> lock(link_list_locks_[id]);
      auto& in_edges = getEdges(id, level);
      in_edges.insert(internal_id);  // 添加反向边
    }
  }

  // 检查反向连接是否一致
  bool checkReverseConnection() {
    int edge_count = 0;
    int reversed_edge_count = 0;
    for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
      for (int level = 0; level <= element_levels_[internal_id]; ++level) {
        unsigned int* data;
        if (level == 0) {
          data = get_linklist0(internal_id);
        } else {
          data = get_linklist(internal_id, level);
        }
        auto link_list = data + 1;
        auto size = getListCount(data);
        edge_count += size;
        reversed_edge_count += getEdges(internal_id, level).size();
        for (int j = 0; j < size; ++j) {
          auto id = link_list[j];
          const auto& in_edges = getEdges(id, level);
          if (in_edges.find(internal_id) == in_edges.end()) {
            std::cout << "can not find internal_id (" << internal_id << ") in its neighbor (" << id << ")" << std::endl;
            return false;
          }
        }
      }
    }

    if (edge_count != reversed_edge_count) {
      std::cout << "mismatch: edge_count (" << edge_count << ") != reversed_edge_count(" << reversed_edge_count << ")"
                << std::endl;
      return false;
    }

    return true;
  }

  // 通过内部 ID 获取数据
  inline char* getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_->GetElementPtr(internal_id, offsetData_));
  }

  // 使用暴力法获取 K 个最近邻
  std::priority_queue<std::pair<float, labeltype>> bruteForce(const void* data_point, int64_t k) override {
    std::priority_queue<std::pair<float, labeltype>> results;
    for (uint32_t i = 0; i < cur_element_count_; i++) {
      float dist = fstdistfunc_(data_point, getDataByInternalId(i), dist_func_param_);
      if (results.size() < k) {
        results.push({dist, *getExternalLabeLp(i)});
      } else {
        float current_max_dist = results.top().first;
        if (dist < current_max_dist) {
          results.pop();
          results.push({dist, *getExternalLabeLp(i)});
        }
      }
    }
    return results;
  }

  // 获取随机层级
  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  // 获取最大元素数
  size_t getMaxElements() override { return max_elements_; }

  // 获取当前元素数
  size_t getCurrentElementCount() override { return cur_element_count_; }

  // 获取已删除元素数
  size_t getDeletedCount() override { return num_deleted_; }

  // searchBaseLayer: 搜索基本层中的最近邻
  // 该函数搜索给定入口点 `ep_id` 和数据点 `data_point` 在给定层次 `layer` 中的最近邻。
  // 它使用优先队列来保存候选节点，并通过逐层扩展找到最近邻。
  std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
  searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
    // 获取一个空闲的访问列表，用于标记节点是否已访问
    auto vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;  // 当前访问标记

    // 创建两个优先队列：一个用于保存最顶端的候选节点，另一个用于扩展候选节点集合
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        top_candidates(allocator_);
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        candidateSet(allocator_);

    float lowerBound;  // 距离的下界

    // 如果入口点没有被标记为删除，计算入口点与数据点的距离
    if (!isMarkedDeleted(ep_id)) {
      float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
      top_candidates.emplace(dist, ep_id);  // 将入口点加入 top_candidates
      lowerBound = dist;                    // 更新下界为该距离
      candidateSet.emplace(-dist, ep_id);   // 将入口点加入候选集合，距离取负值方便最大堆使用
    } else {
      lowerBound = std::numeric_limits<float>::max();  // 如果入口点被删除，则设置最大下界
      candidateSet.emplace(-lowerBound, ep_id);        // 将最大下界加入候选集合
    }

    visited_array[ep_id] = visited_array_tag;  // 标记入口点为已访问

    // 开始扩展候选节点
    while (!candidateSet.empty()) {
      std::pair<float, tableint> curr_el_pair = candidateSet.top();  // 获取当前候选节点
      // 如果候选节点的距离大于下界，且最顶端的候选节点数已经达到 ef_construction，结束搜索
      if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
        break;
      }
      candidateSet.pop();  // 弹出当前节点

      tableint curNodeNum = curr_el_pair.second;  // 当前节点的 ID

      // 加锁，避免多线程竞争
      std::unique_lock<std::recursive_mutex> lock(link_list_locks_[curNodeNum]);

      // 获取当前节点的邻居节点信息
      int* data;
      if (layer == 0) {
        data = (int*)get_linklist0(curNodeNum);  // 获取第 0 层的邻居节点
      } else {
        data = (int*)get_linklist(curNodeNum, layer);  // 获取指定层的邻居节点
      }

      size_t size = getListCount((linklistsizeint*)data);  // 获取邻居数量
      tableint* datal = (tableint*)(data + 1);             // 获取邻居节点 ID 列表

      // 使用 SSE 指令进行预取，提高访问效率
#ifdef USE_SSE
      _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      // 遍历邻居节点
      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);  // 获取当前邻居节点 ID

        // 使用 SSE 指令进行预取
#ifdef USE_SSE
        size_t pre_l = std::min(j, size - 2);
        _mm_prefetch((char*)(visited_array + *(datal + pre_l + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + pre_l + 1)), _MM_HINT_T0);
#endif

        // 如果该邻居已经访问过，则跳过
        if (visited_array[candidate_id] == visited_array_tag) continue;

        visited_array[candidate_id] = visited_array_tag;  // 标记邻居为已访问

        char* currObj1 = (getDataByInternalId(candidate_id));  // 获取当前邻居的数据

        // 计算当前邻居与查询数据点的距离
        float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

        // 如果当前候选集合未满或距离更小，则将邻居加入候选集合
        if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);  // 将邻居加入候选集合

          // 使用 SSE 预取
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

          // 如果该邻居没有被删除，则将其加入最顶端候选集合
          if (!isMarkedDeleted(candidate_id)) top_candidates.emplace(dist1, candidate_id);

          // 如果候选集合超出限制，则弹出最远的节点
          if (top_candidates.size() > ef_construction_) top_candidates.pop();

          // 更新下界
          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
    }

    // 释放访问列表
    visited_list_pool_->releaseVisitedList(vl);

    // 返回最顶端的候选节点
    return top_candidates;
  }

  // searchBaseLayerST: 支持删除标记和过滤器的最近邻搜索
  // 该函数在基础层中进行最近邻搜索，同时支持处理删除标记，并通过过滤器筛选允许的节点。
  // `has_deletions` 表示是否支持删除标记，`collect_metrics` 表示是否收集性能指标。
  // `isIdAllowed` 是一个用于过滤 ID 的可选函数。
  template <bool has_deletions, bool collect_metrics = false>
  std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
  searchBaseLayerST(tableint ep_id, const void* data_point, size_t ef, BaseFilterFunctor* isIdAllowed = nullptr) const {
    // 获取一个空闲的访问列表，用于标记节点是否已访问
    auto vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;  // 当前访问标记

    // 创建两个优先队列：一个用于保存最顶端的候选节点，另一个用于扩展候选节点集合
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        top_candidates(allocator_);
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        candidate_set(allocator_);

    float lowerBound;  // 距离的下界

    // 如果节点未删除，且通过过滤器检查通过，则计算节点与数据点的距离
    if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
      float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
      lowerBound = dist;                    // 更新下界为该距离
      top_candidates.emplace(dist, ep_id);  // 将入口点加入 top_candidates
      candidate_set.emplace(-dist, ep_id);  // 将入口点加入候选集合，距离取负值方便最大堆使用
    } else {
      lowerBound = std::numeric_limits<float>::max();  // 如果节点已删除或不允许，设置最大下界
      candidate_set.emplace(-lowerBound, ep_id);       // 将最大下界加入候选集合
    }

    visited_array[ep_id] = visited_array_tag;  // 标记入口点为已访问

    // 开始扩展候选节点
    while (!candidate_set.empty()) {
      std::pair<float, tableint> current_node_pair = candidate_set.top();  // 获取当前候选节点

      // 如果候选节点的距离大于下界，且最顶端的候选节点数已经达到 ef，或者没有删除和过滤条件，则停止搜索
      if ((-current_node_pair.first) > lowerBound &&
          (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
        break;
      }
      candidate_set.pop();  // 弹出当前节点

      tableint current_node_id = current_node_pair.second;  // 当前节点的 ID

      // 获取当前节点的邻居节点数据
      int* data = (int*)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint*)data);  // 获取邻居数量

      // 如果需要收集性能指标，更新相关计数
      if (collect_metrics) {
        metric_hops_++;                         // 增加跳数（每次扩展邻居时）
        metric_distance_computations_ += size;  // 增加距离计算次数（每个节点的邻居都计算一次距离）
      }

      // 获取邻居节点数据的指针
      auto vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + 1)), offsetData_);

      // 使用 SSE 指令进行预取，提高访问效率
#ifdef USE_SSE
      _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);       // 预取访问数组
      _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);  // 预取下一个访问
      _mm_prefetch(vector_data_ptr, _MM_HINT_T0);                            // 预取数据
      _mm_prefetch((char*)(data + 2), _MM_HINT_T0);                          // 预取邻居数据
#endif

      // 遍历当前节点的所有邻居
      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);        // 获取邻居节点的 ID
        size_t pre_l = std::min(j, size - 2);  // 计算预取位置
        auto vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + pre_l + 1)), offsetData_);

        // 使用 SSE 指令进行预取
#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(data + pre_l + 1)), _MM_HINT_T0);
        _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  // 预取邻居数据
#endif

        // 如果该邻居已经访问过，则跳过
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;  // 标记邻居为已访问

          char* currObj1 = (getDataByInternalId(candidate_id));               // 获取当前邻居的数据
          float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);  // 计算数据点与当前邻居的距离

          // 如果候选集合未满或该邻居距离更小，则将其加入候选集合
          if (top_candidates.size() < ef || lowerBound > dist) {
            candidate_set.emplace(-dist, candidate_id);  // 将邻居加入候选集合
            auto vector_data_ptr = data_level0_memory_->GetElementPtr(candidate_set.top().second, offsetLevel0_);

            // 使用 SSE 预取
#ifdef USE_SSE
            _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
#endif

            // 如果该邻居没有被删除且通过过滤器，则将其加入最顶端候选集合
            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
              top_candidates.emplace(dist, candidate_id);

            // 如果候选集合超出限制，则弹出最远的节点
            if (top_candidates.size() > ef) top_candidates.pop();

            // 更新下界
            if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
          }
        }
      }
    }

    // 释放访问列表
    visited_list_pool_->releaseVisitedList(vl);

    // 返回最顶端的候选节点
    return top_candidates;
  }

  // searchBaseLayerST: 支持删除标记和过滤器的半径内最近邻搜索。
  // 该函数在基础层中进行最近邻搜索，支持处理删除标记、过滤器，并考虑距离上限（radius）和近邻数（ef）。
  // `has_deletions` 表示是否支持删除标记，`collect_metrics` 表示是否收集性能指标。
  // `radius` 表示查询的最大距离，`ef` 为最大近邻数，`isIdAllowed` 是用于节点筛选的可选过滤器函数。
  template <bool has_deletions, bool collect_metrics = false>
  std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
  searchBaseLayerST(tableint ep_id, const void* data_point, float radius, int64_t ef,
                    BaseFilterFunctor* isIdAllowed = nullptr) const {
    // 获取一个空闲的访问列表，用于标记节点是否已访问
    auto vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;  // 当前访问标记

    // 创建两个优先队列：一个用于保存最顶端的候选节点，另一个用于扩展候选节点集合
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        top_candidates(allocator_);
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        candidate_set(allocator_);

    float lowerBound;  // 距离的下界

    // 如果节点未删除，且通过过滤器检查通过，则计算节点与数据点的距离
    if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
      float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
      lowerBound = dist;  // 更新下界为该距离

      // 如果距离小于等于半径加上一个误差阈值，则将该节点加入 top_candidates
      if (dist <= radius + THRESHOLD_ERROR) top_candidates.emplace(dist, ep_id);

      // 将入口点加入候选集合，距离取负值方便最大堆使用
      candidate_set.emplace(-dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<float>::max();  // 如果节点已删除或不允许，设置最大下界
      candidate_set.emplace(-lowerBound, ep_id);       // 将最大下界加入候选集合
    }

    visited_array[ep_id] = visited_array_tag;  // 标记入口点为已访问
    uint64_t visited_count = 0;                // 已访问节点的计数器

    // 开始扩展候选节点
    while (!candidate_set.empty()) {
      std::pair<float, tableint> current_node_pair = candidate_set.top();  // 获取当前候选节点

      candidate_set.pop();  // 弹出当前节点

      tableint current_node_id = current_node_pair.second;  // 当前节点的 ID
      int* data = (int*)get_linklist0(current_node_id);     // 获取当前节点的邻居数据
      size_t size = getListCount((linklistsizeint*)data);   // 获取邻居数量

      // 如果需要收集性能指标，更新相关计数
      if (collect_metrics) {
        metric_hops_++;                         // 增加跳数（每次扩展邻居时）
        metric_distance_computations_ += size;  // 增加距离计算次数（每个节点的邻居都计算一次距离）
      }

      // 获取邻居节点数据的指针
      auto vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + 1)), offsetData_);

      // 使用 SSE 指令进行预取，提高访问效率
#ifdef USE_SSE
      _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);       // 预取访问数组
      _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);  // 预取下一个访问
      _mm_prefetch(vector_data_ptr, _MM_HINT_T0);                            // 预取数据
      _mm_prefetch((char*)(data + 2), _MM_HINT_T0);                          // 预取邻居数据
#endif

      // 遍历当前节点的所有邻居
      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);        // 获取邻居节点的 ID
        size_t pre_l = std::min(j, size - 2);  // 计算预取位置
        auto vector_data_ptr = data_level0_memory_->GetElementPtr((*(data + pre_l + 1)), offsetData_);

        // 使用 SSE 指令进行预取
#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(data + pre_l + 1)), _MM_HINT_T0);
        _mm_prefetch(vector_data_ptr, _MM_HINT_T0);  // 预取邻居数据
#endif

        // 如果该邻居已经访问过，则跳过
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;  // 标记邻居为已访问
          ++visited_count;                                  // 增加已访问节点计数

          char* currObj1 = (getDataByInternalId(candidate_id));               // 获取当前邻居的数据
          float dist = fstdistfunc_(data_point, currObj1, dist_func_param_);  // 计算数据点与当前邻居的距离

          // 如果候选集合未满，或者该邻居的距离小于等于半径加误差阈值，或者该邻居的距离小于当前下界，则将其加入候选集合
          if (visited_count < ef || dist <= radius + THRESHOLD_ERROR || lowerBound > dist) {
            candidate_set.emplace(-dist, candidate_id);  // 将邻居加入候选集合
            auto vector_data_ptr = data_level0_memory_->GetElementPtr(candidate_set.top().second, offsetLevel0_);

            // 使用 SSE 预取
#ifdef USE_SSE
            _mm_prefetch(vector_data_ptr, _MM_HINT_T0);
#endif

            // 如果该邻居没有被删除且通过过滤器，则将其加入最顶端候选集合
            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
              top_candidates.emplace(dist, candidate_id);

            // 更新下界
            if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
          }
        }
      }
    }

    // 从 top_candidates 中删除所有距离大于 radius + THRESHOLD_ERROR 的节点
    while (not top_candidates.empty() && top_candidates.top().first > radius + THRESHOLD_ERROR) {
      top_candidates.pop();
    }

    // 释放访问列表
    visited_list_pool_->releaseVisitedList(vl);

    // 返回最顶端的候选节点（距离小于等于 radius + THRESHOLD_ERROR）
    return top_candidates;
  }

  // getNeighborsByHeuristic2: 该函数使用启发式方法根据优先队列 `top_candidates` 选择 M 个最合适的候选邻居。
  // 它从 `top_candidates` 中获取候选节点并根据距离的启发式规则筛选出最好的 M 个邻居。
  // `top_candidates` 是包含候选节点的优先队列，`M` 是我们希望筛选出的邻居的数量。
  void getNeighborsByHeuristic2(
      std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>&
          top_candidates,
      const size_t M) {
    // 如果当前候选节点数小于 M，则无需筛选，直接返回
    if (top_candidates.size() < M) {
      return;
    }

    // 创建一个优先队列，用于存储最接近的候选节点
    std::priority_queue<std::pair<float, tableint>> queue_closest;

    // 创建一个返回列表，用于存储最终选择的 M 个邻居
    vsag::Vector<std::pair<float, tableint>> return_list(allocator_);

    // 将 `top_candidates` 中的元素按顺序添加到 `queue_closest`，并移除原队列中的元素
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
      top_candidates.pop();
    }

    // 开始筛选出最合适的 M 个邻居
    while (queue_closest.size()) {
      // 如果已经选择了 M 个邻居，则停止
      if (return_list.size() >= M) break;

      // 获取当前候选节点并移除它
      std::pair<float, tableint> curent_pair = queue_closest.top();
      float floato_query = -curent_pair.first;  // 获取当前节点的距离（取负值恢复原值）
      queue_closest.pop();

      // 假设当前候选是一个合适的节点
      bool good = true;

      // 遍历已经选中的邻居，确保当前候选节点与所有已选邻居的距离都大于当前节点的距离
      for (std::pair<float, tableint> second_pair : return_list) {
        // 计算当前候选节点与已选节点的距离
        float curdist = fstdistfunc_(getDataByInternalId(second_pair.second), getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
        // 如果当前候选与某个已选邻居的距离小于当前候选的距离，说明当前候选不符合要求
        if (curdist < floato_query) {
          good = false;  // 标记为不符合要求
          break;
        }
      }

      // 如果当前候选节点符合要求，则将其加入返回列表
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    // 将最终选出的邻居重新添加到 `top_candidates` 中
    for (std::pair<float, tableint> curent_pair : return_list) {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  // get_linklist0: 返回一个指向某个元素在第 0 层链接列表的指针
  linklistsizeint* get_linklist0(tableint internal_id) const {
    return (linklistsizeint*)(data_level0_memory_->GetElementPtr(internal_id, offsetLevel0_));
  }

  // get_linklist: 返回某个元素在指定层级 `level` 的链接列表指针
  linklistsizeint* get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint*)(link_lists_[internal_id] + (level - 1) * size_links_per_element_);
  }

  // get_linklist_at_level: 根据指定的层级返回链接列表。若层级为 0，则调用 `get_linklist0`，否则调用 `get_linklist`
  linklistsizeint* get_linklist_at_level(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
  }

  // mutuallyConnectNewElement：该函数将当前元素 `cur_c` 与候选的最接近元素建立连接，
  // 通过启发式方法选择最合适的邻居，并更新相关连接信息。
  // `data_point` 是当前元素的数据点，`cur_c` 是当前元素的内部 ID，`top_candidates` 是候选节点的优先队列，
  // `level` 是当前的层级，`isUpdate` 表示是否为更新操作。
  tableint mutuallyConnectNewElement(
      const void* data_point, tableint cur_c,
      std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>&
          top_candidates,
      int level, bool isUpdate) {
    // 获取当前层级的最大邻居数 `m_curmax`，根据 `level` 选择不同的最大邻居数
    size_t m_curmax = level ? maxM_ : maxM0_;

    // 使用启发式方法获取最接近的 M 个邻居
    getNeighborsByHeuristic2(top_candidates, M_);

    // 如果启发式返回的候选节点数大于 M，抛出异常
    if (top_candidates.size() > M_)
      throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    // 创建一个 `selectedNeighbors` 向量来存储选择的邻居
    vsag::Vector<tableint> selectedNeighbors(allocator_);
    selectedNeighbors.reserve(M_);

    // 从优先队列中依次取出选定的邻居节点，并存储到 `selectedNeighbors` 中
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    // 获取当前选择的最近邻元素（最后一个节点）
    tableint next_closest_entry_point = selectedNeighbors.back();

    {
      // 如果是更新操作，则加锁以确保更新时的线程安全
      std::unique_lock<std::recursive_mutex> lock(link_list_locks_[cur_c], std::defer_lock);
      if (isUpdate) {
        lock.lock();  // 在更新时锁定当前元素
      }

      // 更新当前元素与选择的邻居的连接关系
      updateConnections(cur_c, selectedNeighbors, level, isUpdate);
    }

    // 遍历每个选定的邻居，更新它们与当前元素的连接关系
    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      // 对于每个邻居，加锁以确保线程安全
      std::unique_lock<std::recursive_mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint* ll_other;
      // 根据层级选择对应的链接列表
      if (level == 0)
        ll_other = get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = get_linklist(selectedNeighbors[idx], level);

      // 获取该邻居的链接列表的大小
      size_t sz_link_list_other = getListCount(ll_other);

      // 检查链接列表的大小是否合法，不能大于最大允许的邻居数
      if (sz_link_list_other > m_curmax) throw std::runtime_error("Bad value of sz_link_list_other");

      // 如果尝试将元素连接到自己，抛出异常
      if (selectedNeighbors[idx] == cur_c) throw std::runtime_error("Trying to connect an element to itself");

      // 检查是否尝试连接到一个不存在的层级
      if (level > element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error("Trying to make a link on a non-existent level");

      // 获取该邻居的连接数据
      tableint* data = (tableint*)(ll_other + 1);

      bool is_cur_c_present = false;
      // 如果是更新操作，则检查当前元素是否已存在于邻居的连接列表中
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // 如果当前元素已存在于邻居的连接列表中，则无需修改连接
      if (!is_cur_c_present) {
        // 如果邻居的连接列表未满，则将当前元素添加到邻居的连接列表中
        if (sz_link_list_other < m_curmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);

          // 如果启用了反向边，则更新当前元素的反向连接
          if (use_reversed_edges_) {
            auto& cur_in_edges = getEdges(cur_c, level);
            cur_in_edges.insert(selectedNeighbors[idx]);
          }
        } else {
          // 如果邻居的连接列表已满，则通过启发式方法找出“最弱”的邻居并替换
          float d_max =
              fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);

          // 启发式方法：计算当前元素与所有邻居的距离，挑选出合适的替换
          std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
              candidates(allocator_);
          candidates.emplace(d_max, cur_c);

          // 遍历邻居，计算它们的距离，并加入候选队列
          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                            dist_func_param_),
                               data[j]);
          }

          // 使用启发式方法获取最接近的 M 个邻居
          getNeighborsByHeuristic2(candidates, m_curmax);

          // 将最接近的邻居存储到 `cand_neighbors` 中
          vsag::Vector<tableint> cand_neighbors(allocator_);
          while (candidates.size() > 0) {
            cand_neighbors.push_back(candidates.top().second);
            candidates.pop();
          }

          // 更新邻居的连接关系
          updateConnections(selectedNeighbors[idx], cand_neighbors, level, true);
        }
      }
    }

    // 返回下一个最接近的入口点
    return next_closest_entry_point;
  }

  // resizeIndex：该函数用于调整索引的大小，以便容纳更多的元素。
  // `new_max_elements` 是新的最大元素数量。
  void resizeIndex(size_t new_max_elements) override {
    // 检查新大小是否小于当前元素数量，如果是则抛出异常
    if (new_max_elements < cur_element_count_)
      throw std::runtime_error("Cannot Resize, max element is less than the current number of elements");

    // 重新分配访问列表池
    visited_list_pool_.reset(new VisitedListPool(1, new_max_elements, allocator_));

    // 重新分配 `element_levels_` 数组
    auto element_levels_new = (int*)allocator_->Reallocate(element_levels_, new_max_elements * sizeof(int));
    if (element_levels_new == nullptr) {
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate element_levels_");
    }
    element_levels_ = element_levels_new;

    // 重新分配链接列表锁
    vsag::Vector<std::recursive_mutex>(new_max_elements, allocator_).swap(link_list_locks_);

    // 如果启用了归一化，则重新分配 `molds_` 数组
    if (normalize_) {
      auto new_molds = (float*)allocator_->Reallocate(molds_, new_max_elements * sizeof(float));
      if (new_molds == nullptr) {
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate molds_");
      }
      molds_ = new_molds;
    }

    // 重新分配基础层内存
    if (not data_level0_memory_->Resize(new_max_elements))
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");

    // 如果启用了反向边，则重新分配反向链接列表内存
    if (use_reversed_edges_) {
      auto reversed_level0_link_list_new = (reverselinklist**)allocator_->Reallocate(
          reversed_level0_link_list_, new_max_elements * sizeof(reverselinklist*));
      if (reversed_level0_link_list_new == nullptr) {
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate reversed_level0_link_list_");
      }
      reversed_level0_link_list_ = reversed_level0_link_list_new;

      // 初始化新分配的内存
      memset(reversed_level0_link_list_ + max_elements_, 0,
             (new_max_elements - max_elements_) * sizeof(reverselinklist*));

      auto reversed_link_lists_new = (vsag::UnorderedMap<int, reverselinklist>**)allocator_->Reallocate(
          reversed_link_lists_, new_max_elements * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
      if (reversed_link_lists_new == nullptr) {
        throw std::runtime_error("Not enough memory: resizeIndex failed to allocate reversed_link_lists_");
      }
      reversed_link_lists_ = reversed_link_lists_new;
      memset(reversed_link_lists_ + max_elements_, 0,
             (new_max_elements - max_elements_) * sizeof(std::map<int, reverselinklist>*));
    }

    // 重新分配其他层的链接列表
    char** linkLists_new = (char**)allocator_->Reallocate(link_lists_, sizeof(void*) * new_max_elements);
    if (linkLists_new == nullptr)
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    link_lists_ = linkLists_new;
    memset(link_lists_ + max_elements_, 0, (new_max_elements - max_elements_) * sizeof(void*));

    max_elements_ = new_max_elements;
  }

  // calcSerializeSize：该函数计算序列化对象所需的总字节数。
  // 返回值：返回计算出的总字节数。
  size_t calcSerializeSize() override {
    size_t size = 0;

    // 计算每个数据成员的大小，并累加到总字节数
    size += sizeof(offsetLevel0_);           // offsetLevel0_ 字段占用的字节数
    size += sizeof(max_elements_);           // max_elements_ 字段占用的字节数
    size += sizeof(cur_element_count_);      // cur_element_count_ 字段占用的字节数
    size += sizeof(size_data_per_element_);  // size_data_per_element_ 字段占用的字节数
    size += sizeof(label_offset_);           // label_offset_ 字段占用的字节数
    size += sizeof(offsetData_);             // offsetData_ 字段占用的字节数
    size += sizeof(maxlevel_);               // maxlevel_ 字段占用的字节数
    size += sizeof(enterpoint_node_);        // enterpoint_node_ 字段占用的字节数
    size += sizeof(maxM_);                   // maxM_ 字段占用的字节数

    size += sizeof(maxM0_);            // maxM0_ 字段占用的字节数
    size += sizeof(M_);                // M_ 字段占用的字节数
    size += sizeof(mult_);             // mult_ 字段占用的字节数
    size += sizeof(ef_construction_);  // ef_construction_ 字段占用的字节数

    // 累加 data_level0_memory_ 所占用的字节数
    size += data_level0_memory_->GetSize();  // 计算 data_level0_memory_ 内存的字节数

    // 累加与每个元素相关的链接列表大小
    size += maxM0_ * sizeof(uint32_t) * max_elements_;  // maxM0_ * uint32_t 大小为每个元素的链接列表大小

    // 遍历每个元素，累加它的链接列表大小
    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int link_list_size = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      size += sizeof(link_list_size);  // 链接列表大小本身
      if (link_list_size) {
        size += link_list_size * 2;  // 计算并加上反向链接列表的大小
      }
    }

    // 如果启用了归一化，累加 molds_ 数组的大小
    if (normalize_) {
      size += max_elements_ * sizeof(float);  // molds_ 数组的总大小
    }

    return size;  // 返回总的序列化大小
  }

  // saveIndex：将索引数据保存到指定的内存区域。
  void saveIndex(void* d) override {
    char* dest = (char*)d;            // 转换目标内存地址为 char* 类型
    BufferStreamWriter writer(dest);  // 创建一个缓冲流写入器
    SerializeImpl(writer);            // 执行序列化
  }

  // saveIndex：将索引数据保存到输出流。
  void saveIndex(std::ostream& out_stream) override {
    IOStreamWriter writer(out_stream);  // 创建一个 IO 流写入器
    SerializeImpl(writer);              // 执行序列化
  }

  // saveIndex：将索引数据保存到指定文件路径。
  void saveIndex(const std::string& location) override {
    std::ofstream output(location, std::ios::binary);  // 打开文件流以二进制模式写入
    IOStreamWriter writer(output);                     // 创建一个 IO 流写入器
    SerializeImpl(writer);                             // 执行序列化
    output.close();                                    // 关闭文件流
  }

  // WriteOne：将一个对象 `value` 序列化并写入到流中。
  // `T` 是对象类型，`writer` 是流写入器，`value` 是要序列化的对象。
  template <typename T>
  static void WriteOne(StreamWriter& writer, T& value) {
    writer.Write(reinterpret_cast<char*>(&value), sizeof(value));  // 将对象按字节写入流
  }

  // SerializeImpl：执行实际的序列化操作，将所有需要保存的数据写入流。
  void SerializeImpl(StreamWriter& writer) {
    // 按顺序写入每个数据成员
    WriteOne(writer, offsetLevel0_);
    WriteOne(writer, max_elements_);
    WriteOne(writer, cur_element_count_);
    WriteOne(writer, size_data_per_element_);
    WriteOne(writer, label_offset_);
    WriteOne(writer, offsetData_);
    WriteOne(writer, maxlevel_);
    WriteOne(writer, enterpoint_node_);
    WriteOne(writer, maxM_);

    WriteOne(writer, maxM0_);
    WriteOne(writer, M_);
    WriteOne(writer, mult_);
    WriteOne(writer, ef_construction_);

    // 序列化 `data_level0_memory_` 的内容
    data_level0_memory_->SerializeImpl(writer, cur_element_count_);

    // 序列化每个元素的链接列表
    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int link_list_size = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      WriteOne(writer, link_list_size);  // 写入每个元素的链接列表大小
      if (link_list_size) {
        writer.Write(link_lists_[i], link_list_size);  // 写入实际的链接数据
      }
    }

    // 如果启用了归一化，写入 molds_ 数据
    if (normalize_) {
      writer.Write(reinterpret_cast<char*>(molds_), max_elements_ * sizeof(float));  // 写入 molds_ 数组
    }
  }

  // loadIndex：使用提供的读函数加载索引数据。
  // `read_func` 是读取数据的回调函数，`s` 是空间接口，`max_elements_i` 是最大元素数量。
  void loadIndex(std::function<void(uint64_t, uint64_t, void*)> read_func, SpaceInterface* s,
                 size_t max_elements_i) override {
    int64_t cursor = 0;                              // 游标初始化为 0，用于标记读取位置
    ReadFuncStreamReader reader(read_func, cursor);  // 创建流读取器
    DeserializeImpl(reader, s, max_elements_i);      // 调用反序列化函数加载数据
  }

  // loadIndex：从文件流中加载索引数据。
  // `in_stream` 是输入流，`s` 是空间接口，`max_elements_i` 是最大元素数量。
  void loadIndex(std::istream& in_stream, SpaceInterface* s, size_t max_elements_i) override {
    IOStreamReader reader(in_stream);                  // 创建文件流读取器
    this->DeserializeImpl(reader, s, max_elements_i);  // 调用反序列化函数加载数据
  }

  // loadIndex：从指定路径的文件加载索引数据。
  // `location` 是文件路径，`s` 是空间接口，`max_elements_i` 是最大元素数量，默认为 0。
  void loadIndex(const std::string& location, SpaceInterface* s, size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);   // 打开文件流，读取二进制数据
    IOStreamReader reader(input);                      // 创建文件流读取器
    this->DeserializeImpl(reader, s, max_elements_i);  // 调用反序列化函数加载数据
    input.close();                                     // 关闭文件流
  }

  // ReadOne：从流中读取一个对象并填充到 `value` 中。
  // `reader` 是流读取器，`value` 是要读取的对象。
  template <typename T>
  static void ReadOne(StreamReader& reader, T& value) {
    reader.Read(reinterpret_cast<char*>(&value), sizeof(value));  // 从流中读取数据
  }

  // DeserializeImpl：执行实际的反序列化操作，将存储的数据恢复为对象。
  // `reader` 是流读取器，`s` 是空间接口，`max_elements_i` 是最大元素数量，默认为 0。
  void DeserializeImpl(StreamReader& reader, SpaceInterface* s, size_t max_elements_i = 0) {
    ReadOne(reader, offsetLevel0_);  // 读取 offsetLevel0_

    size_t max_elements;
    ReadOne(reader, max_elements);                          // 读取最大元素数量
    max_elements = std::max(max_elements, max_elements_i);  // 确保不小于 max_elements_i
    max_elements = std::max(max_elements, max_elements_);   // 确保不小于当前的最大元素数量

    // 读取其他数据成员
    ReadOne(reader, cur_element_count_);
    ReadOne(reader, size_data_per_element_);
    ReadOne(reader, label_offset_);
    ReadOne(reader, offsetData_);
    ReadOne(reader, maxlevel_);
    ReadOne(reader, enterpoint_node_);

    // 读取并恢复其他参数
    ReadOne(reader, maxM_);
    ReadOne(reader, maxM0_);
    ReadOne(reader, M_);
    ReadOne(reader, mult_);
    ReadOne(reader, ef_construction_);

    // 获取空间接口提供的参数
    data_size_ = s->get_data_size();              // 数据大小
    fstdistfunc_ = s->get_dist_func();            // 距离计算函数
    dist_func_param_ = s->get_dist_func_param();  // 距离函数参数

    // 根据最大元素数量调整索引大小
    resizeIndex(max_elements);

    // 反序列化 data_level0_memory_ 中的数据
    data_level0_memory_->DeserializeImpl(reader, cur_element_count_);

    // 设置链接列表的大小
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    // 设置 level0 链接列表的大小
    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

    // 初始化链接列表锁和标签操作锁
    vsag::Vector<std::recursive_mutex>(max_elements, allocator_).swap(link_list_locks_);
    vsag::Vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS, allocator_).swap(label_op_locks_);

    // 计算反向边缘的大小（乘以倒数）
    revSize_ = 1.0 / mult_;

    // 处理每个元素的链接列表
    for (size_t i = 0; i < cur_element_count_; i++) {
      label_lookup_[getExternalLabel(i)] = i;  // 根据标签查找元素
      unsigned int link_list_size;
      ReadOne(reader, link_list_size);  // 读取链接列表的大小
      if (link_list_size == 0) {
        element_levels_[i] = 0;    // 如果没有链接，设置元素级别为 0
        link_lists_[i] = nullptr;  // 链接列表为空
      } else {
        // 计算链接列表的层级，并分配内存
        element_levels_[i] = link_list_size / size_links_per_element_;
        link_lists_[i] = (char*)allocator_->Allocate(link_list_size);
        if (link_lists_[i] == nullptr)
          throw std::runtime_error(
              "Not enough memory: loadIndex failed to allocate linklist");  // 如果内存分配失败，抛出异常
        reader.Read(link_lists_[i], link_list_size);                        // 读取链接列表数据
      }
    }

    // 如果启用了归一化，读取 molds_ 数组
    if (normalize_) {
      reader.Read(reinterpret_cast<char*>(molds_), max_elements_ * sizeof(float));
    }

    // 如果使用反向边缘，更新每个元素的反向边缘信息
    if (use_reversed_edges_) {
      for (int internal_id = 0; internal_id < cur_element_count_; ++internal_id) {
        for (int level = 0; level <= element_levels_[internal_id]; ++level) {
          unsigned int* data = get_linklist_at_level(internal_id, level);
          auto link_list = data + 1;
          auto size = getListCount(data);
          for (int j = 0; j < size; ++j) {
            auto id = link_list[j];
            auto& in_edges = getEdges(id, level);
            in_edges.insert(internal_id);  // 插入反向边缘
          }
        }
      }
    }

    // 检查每个元素是否已被标记为删除
    for (size_t i = 0; i < cur_element_count_; i++) {
      if (isMarkedDeleted(i)) {
        num_deleted_ += 1;                                        // 统计已删除元素
        if (allow_replace_deleted_) deleted_elements_.insert(i);  // 如果允许替换已删除元素，将其插入删除元素集合
      }
    }
  }

  // getDataByLabel：通过标签获取元素数据。
  // `label` 是标签值，返回对应的数据指针。
  const float* getDataByLabel(labeltype label) const override {
    std::lock_guard<std::mutex> lock_label(getLabelOpMutex(label));  // 获取标签操作锁

    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);  // 获取标签查找锁
    auto search = label_lookup_.find(label);                      // 查找标签对应的内部 ID
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
      throw std::runtime_error("Label not found");  // 如果未找到或已删除，抛出异常
    }
    tableint internalId = search->second;
    lock_table.unlock();  // 释放标签查找锁

    char* data_ptrv = getDataByInternalId(internalId);  // 获取内部 ID 对应的数据
    float* data_ptr = (float*)data_ptrv;                // 转换为 float 指针

    return data_ptr;  // 返回数据指针
  }

  // markDelete：标记一个元素为已删除状态，不会改变当前图的结构。
  // `label` 是要删除的元素的标签。
  void markDelete(labeltype label) {
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));  // 获取标签操作锁

    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);  // 获取标签查找锁
    auto search = label_lookup_.find(label);                      // 查找标签对应的内部 ID
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");  // 如果标签未找到，抛出异常
    }
    tableint internalId = search->second;
    label_lookup_.erase(search);      // 删除标签与内部 ID 的映射
    lock_table.unlock();              // 释放标签查找锁
    markDeletedInternal(internalId);  // 标记元素为已删除
  }

  /*
   * markDeletedInternal：标记内部元素为已删除状态。
   * 使用内存的最后 16 位存储链接列表大小，同时标记删除。
   * maxM0_ 限制为低 16 位，但在大多数情况下仍然足够大。
   */
  void markDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count_);  // 确保元素 ID 小于当前元素总数
    if (!isMarkedDeleted(internalId)) {       // 如果元素未被标记为已删除
      unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;  // 获取链接列表指针
      *ll_cur |= DELETE_MARK;        // 将删除标记设置到链接列表的最后两位
      num_deleted_ += 1;             // 增加已删除元素计数
      if (allow_replace_deleted_) {  // 如果允许替换已删除的元素
        std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock_);
        deleted_elements_.insert(internalId);  // 将已删除元素插入删除集合
      }
    } else {
      throw std::runtime_error("The requested to delete element is already deleted");  // 如果已经标记为删除，抛出异常
    }
  }

  /*
   * unmarkDelete：移除元素的删除标记，不改变图的当前结构。
   * 注意：如果启用了替换已删除元素的功能，则不安全，
   * 因为标记为已删除的元素可能会被 addPoint 完全删除。
   */
  void unmarkDelete(labeltype label) {
    // 锁定与标签相关的所有操作
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
    auto search = label_lookup_.find(label);  // 查找标签对应的内部 ID
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");  // 如果标签未找到，抛出异常
    }
    tableint internalId = search->second;
    lock_table.unlock();

    unmarkDeletedInternal(internalId);  // 移除删除标记
  }

  /*
   * unmarkDeletedInternal：移除元素的删除标记。
   */
  void unmarkDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count_);  // 确保元素 ID 小于当前元素总数
    if (isMarkedDeleted(internalId)) {        // 如果元素已被标记为删除
      unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;  // 获取链接列表指针
      *ll_cur &= ~DELETE_MARK;                                                  // 清除删除标记
      num_deleted_ -= 1;                                                        // 减少已删除元素计数
      if (allow_replace_deleted_) {                                             // 如果允许替换已删除的元素
        std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock_);
        deleted_elements_.erase(internalId);  // 从删除集合中移除该元素
      }
    } else {
      throw std::runtime_error("The requested to undelete element is not deleted");  // 如果元素没有被删除，抛出异常
    }
  }

  /*
   * isMarkedDeleted：检查元素是否已被标记为删除。
   */
  bool isMarkedDeleted(tableint internalId) const {
    unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;  // 获取链接列表指针
    return *ll_cur & DELETE_MARK;                                             // 检查删除标记是否存在
  }

  // 获取链接列表的大小
  unsigned short int getListCount(linklistsizeint* ptr) const {
    return *((unsigned short int*)ptr);  // 返回链接列表的元素数量
  }

  // 设置链接列表的大小
  void setListCount(linklistsizeint* ptr, unsigned short int size) const {
    *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);  // 设置链接列表的元素数量
  }

  /*
   * addPoint：添加一个新的点（元素）。
   * `data_point` 是数据点，`label` 是标签。
   */
  bool addPoint(const void* data_point, labeltype label) override {
    std::lock_guard<std::mutex> lock_label(getLabelOpMutex(label));  // 锁定标签操作
    if (addPoint(data_point, label, -1) == -1) {  // 调用实际的添加函数，-1 表示没有指定 ID
      return false;                               // 如果添加失败，返回 false
    }
    return true;  // 添加成功，返回 true
  }

  /*
   * modify_out_edge：修改元素的输出边（out edge）。
   * `old_internal_id` 是原始内部 ID，`new_internal_id` 是新的内部 ID。
   */
  inline void modify_out_edge(tableint old_internal_id, tableint new_internal_id) {
    for (int level = 0; level <= element_levels_[old_internal_id]; ++level) {
      auto& edges = getEdges(old_internal_id, level);       // 获取旧元素的输出边
      for (const auto& in_node : edges) {                   // 遍历每条输出边
        auto data = get_linklist_at_level(in_node, level);  // 获取连接的邻居节点
        size_t link_size = getListCount(data);              // 获取链接列表的大小
        tableint* links = (tableint*)(data + 1);            // 获取链接数组
        for (int i = 0; i < link_size; ++i) {
          if (links[i] == old_internal_id) {  // 找到旧元素，替换为新元素
            links[i] = new_internal_id;
            break;
          }
        }
      }
    }
  }

  /*
   * modify_in_edges：修改元素的输入边（in edge）。
   * `right_internal_id` 是右边元素的内部 ID，
   * `wrong_internal_id` 是错误的内部 ID，`is_erase` 表示是否删除边。
   */
  inline void modify_in_edges(tableint right_internal_id, tableint wrong_internal_id, bool is_erase) {
    for (int level = 0; level <= element_levels_[right_internal_id]; ++level) {
      auto data = get_linklist_at_level(right_internal_id, level);  // 获取右边元素的链接列表
      size_t link_size = getListCount(data);                        // 获取链接列表的大小
      tableint* links = (tableint*)(data + 1);                      // 获取链接数组
      for (int i = 0; i < link_size; ++i) {
        auto& in_egdes = getEdges(links[i], level);  // 获取输入边
        if (is_erase) {
          in_egdes.erase(wrong_internal_id);  // 如果是删除，移除错误的边
        } else {
          in_egdes.insert(right_internal_id);  // 否则，添加正确的边
        }
      }
    }
  }

  /*
   * swapConnections：交换两个元素的连接关系。
   * `pre_internal_id` 和 `post_internal_id` 分别是前后两个元素的内部 ID。
   */
  bool swapConnections(tableint pre_internal_id, tableint post_internal_id) {
    {
      // 修改图中的连接关系：交换前后元素的连接
      modify_out_edge(pre_internal_id, post_internal_id);  // 修改输出边
      modify_out_edge(post_internal_id, pre_internal_id);  // 修改输出边

      // 交换元素的数据和邻接列表
      auto tmp_data_element = std::shared_ptr<char[]>(new char[size_data_per_element_]);  // 创建临时数据元素
      memcpy(tmp_data_element.get(), get_linklist0(pre_internal_id), size_data_per_element_);  // 复制前元素数据
      memcpy(get_linklist0(pre_internal_id), get_linklist0(post_internal_id),
             size_data_per_element_);  // 复制后元素数据
      memcpy(get_linklist0(post_internal_id), tmp_data_element.get(),
             size_data_per_element_);  // 将前元素数据复制到后元素

      if (normalize_) {
        std::swap(molds_[pre_internal_id], molds_[post_internal_id]);  // 交换归一化数据
      }
      std::swap(link_lists_[pre_internal_id], link_lists_[post_internal_id]);          // 交换链接列表
      std::swap(element_levels_[pre_internal_id], element_levels_[post_internal_id]);  // 交换元素层级
    }

    {
      // 修复反向边缘，避免交换时丢失连接关系
      std::swap(reversed_level0_link_list_[pre_internal_id], reversed_level0_link_list_[post_internal_id]);
      std::swap(reversed_link_lists_[pre_internal_id], reversed_link_lists_[post_internal_id]);

      // 删除不正确的反向连接，并重新插入正确的连接
      modify_in_edges(pre_internal_id, post_internal_id, true);
      modify_in_edges(post_internal_id, pre_internal_id, true);
      modify_in_edges(pre_internal_id, post_internal_id, false);
      modify_in_edges(post_internal_id, pre_internal_id, false);
    }

    // 如果进入点是交换的元素之一，更新进入点
    if (enterpoint_node_ == post_internal_id) {
      enterpoint_node_ = pre_internal_id;
    } else if (enterpoint_node_ == pre_internal_id) {
      enterpoint_node_ = post_internal_id;
    }

    return true;  // 返回 true 表示交换成功
  }

  void dealNoInEdge(tableint id, int level, int m_curmax, int skip_c) {
    // Establish edges from the neighbors of the id pointing to the id.
    auto alone_data = get_linklist_at_level(id, level);
    int alone_size = getListCount(alone_data);
    auto alone_link = (unsigned int*)(alone_data + 1);
    auto& in_edges = getEdges(id, level);
    for (int j = 0; j < alone_size; ++j) {
      if (alone_link[j] == skip_c) {
        continue;
      }
      auto to_edge_data_cur = (unsigned int*)get_linklist_at_level(alone_link[j], level);
      int to_edge_size_cur = getListCount(to_edge_data_cur);
      auto to_edge_data_link_cur = (unsigned int*)(to_edge_data_cur + 1);
      if (to_edge_size_cur < m_curmax) {
        to_edge_data_link_cur[to_edge_size_cur] = id;
        setListCount(to_edge_data_cur, to_edge_size_cur + 1);
        in_edges.insert(alone_link[j]);
      }
    }
  }

  // 删除指定标签的元素
  void removePoint(labeltype label) {
    tableint cur_c = 0;
    tableint internal_id = 0;

    std::lock_guard<std::mutex> lock(global_);  // 锁定全局资源，保证线程安全

    {
      // 查找标签对应的元素，并与最后一个元素交换位置，填补删除元素的位置
      std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
      auto iter = label_lookup_.find(label);  // 查找标签
      if (iter == label_lookup_.end()) {
        throw std::runtime_error("no label in FreshHnsw");  // 如果标签不存在，抛出异常
      } else {
        internal_id = iter->second;  // 获取元素的内部 ID
        label_lookup_.erase(iter);   // 从标签映射中移除该标签
      }

      cur_element_count_--;        // 减少当前元素计数
      cur_c = cur_element_count_;  // 更新当前元素的 ID

      if (cur_c == 0) {
        // 如果删除后没有元素，清空所有连接关系，重置入口节点和最大层级
        for (int level = 0; level < element_levels_[cur_c]; ++level) {
          getEdges(cur_c, level).clear();
        }
        enterpoint_node_ = -1;
        maxlevel_ = -1;
        return;
      } else if (cur_c != internal_id) {
        // 如果当前元素不是要删除的元素，交换位置
        label_lookup_[getExternalLabel(cur_c)] = internal_id;
        swapConnections(cur_c, internal_id);
      }
    }

    // 如果要删除的节点是入口节点，寻找新的入口节点
    if (cur_c == enterpoint_node_) {
      for (int level = maxlevel_; level >= 0; level--) {
        auto data = (unsigned int*)get_linklist_at_level(enterpoint_node_, level);
        int size = getListCount(data);
        if (size != 0) {
          maxlevel_ = level;
          enterpoint_node_ = *(data + 1);  // 设置新的入口节点
          break;
        }
      }
    }

    // 修复删除节点后其他节点的连接关系
    for (int level = 0; level <= element_levels_[cur_c]; ++level) {
      const auto in_edges_cur = getEdges(cur_c, level);
      auto data_cur = get_linklist_at_level(cur_c, level);
      int size_cur = getListCount(data_cur);
      auto data_link_cur = (unsigned int*)(data_cur + 1);

      for (const auto in_edge : in_edges_cur) {
        std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
            candidates(allocator_);
        vsag::UnorderedSet<tableint> unique_ids(allocator_);

        // 将输入节点的邻居添加到候选队列
        for (int i = 0; i < size_cur; ++i) {
          if (data_link_cur[i] == cur_c || data_link_cur[i] == in_edge) {
            continue;
          }
          unique_ids.insert(data_link_cur[i]);
          candidates.emplace(
              fstdistfunc_(getDataByInternalId(data_link_cur[i]), getDataByInternalId(in_edge), dist_func_param_),
              data_link_cur[i]);
        }

        // 将待删除节点的邻居也加入候选队列
        auto in_edge_data_cur = (unsigned int*)get_linklist_at_level(in_edge, level);
        int in_edge_size_cur = getListCount(in_edge_data_cur);
        auto in_edge_data_link_cur = (unsigned int*)(in_edge_data_cur + 1);
        for (int i = 0; i < in_edge_size_cur; ++i) {
          if (in_edge_data_link_cur[i] == cur_c || unique_ids.find(in_edge_data_link_cur[i]) != unique_ids.end()) {
            continue;
          }
          unique_ids.insert(in_edge_data_link_cur[i]);
          candidates.emplace(fstdistfunc_(getDataByInternalId(in_edge_data_link_cur[i]), getDataByInternalId(in_edge),
                                          dist_func_param_),
                             in_edge_data_link_cur[i]);
        }

        if (candidates.size() == 0) {
          setListCount(in_edge_data_cur, 0);      // 如果没有候选项，清空链接
          getEdges(cur_c, level).erase(in_edge);  // 删除当前元素的边
          continue;
        }
        // 连接新元素
        mutuallyConnectNewElement(getDataByInternalId(in_edge), in_edge, candidates, level, true);

        // 处理删除点导致的一些节点没有输入边的情况，并进行修复
        size_t m_curmax = level ? maxM_ : maxM0_;
        for (auto id : unique_ids) {
          if (getEdges(id, level).size() == 0) {
            dealNoInEdge(id, level, m_curmax, cur_c);
          }
        }
      }

      // 移除已删除节点的输出边
      for (int i = 0; i < size_cur; ++i) {
        getEdges(data_link_cur[i], level).erase(cur_c);
      }
    }
  }

  // 添加一个新的点（元素）
  tableint addPoint(const void* data_point, labeltype label, int level) {
    tableint cur_c = 0;

    {
      // 检查是否已经有相同标签的元素，如果有则返回-1
      std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) {
        return -1;  // 如果标签已存在，则不添加新元素
      }

      // 如果当前元素数量超出限制，进行扩展
      if (cur_element_count_ >= max_elements_) {
        resizeIndex(max_elements_ + data_element_per_block_);
      }

      cur_c = cur_element_count_;    // 设置当前元素 ID
      cur_element_count_++;          // 增加元素计数
      label_lookup_[label] = cur_c;  // 将标签与元素 ID 关联
    }

    std::shared_ptr<float[]> normalize_data;
    normalize_vector(data_point, normalize_data);  // 对数据进行归一化处理

    std::unique_lock<std::recursive_mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);  // 获取随机层级
    if (level > 0) curlevel = level;       // 如果传入层级大于0，使用传入的层级

    element_levels_[cur_c] = curlevel;  // 设置元素的层级
    std::unique_lock<std::mutex> lock(global_);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) lock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_->GetElementPtr(cur_c, offsetLevel0_), 0, size_data_per_element_);  // 初始化数据

    // 初始化数据和标签
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    // 如果当前层级大于0，分配新的链接列表
    if (curlevel) {
      auto new_link_lists = (char*)allocator_->Reallocate(link_lists_[cur_c], size_links_per_element_ * curlevel + 1);
      if (new_link_lists == nullptr)
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      link_lists_[cur_c] = new_link_lists;
      memset(link_lists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      // 如果元素已经存在，尝试找到最佳的连接
      if (curlevel < maxlevelcopy) {
        float curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int* data;
            std::unique_lock<std::recursive_mutex> lock(link_list_locks_[currObj]);
            data = get_linklist(currObj, level);
            int size = getListCount(data);

            tableint* datal = (tableint*)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];
              if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
              float d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }

      // 如果入口节点已删除，重新进行候选连接
      bool epDeleted = isMarkedDeleted(enterpoint_copy);
      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
        if (level > maxlevelcopy || level < 0)  // 检查层级是否合理
          throw std::runtime_error("Level error");

        std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
            top_candidates = searchBaseLayer(currObj, data_point, level);
        if (epDeleted) {
          top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_),
                                 enterpoint_copy);
          if (top_candidates.size() > ef_construction_) top_candidates.pop();
        }
        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
      }
    } else {
      // 对于第一个元素，不需要连接
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }

    // 更新入口节点和最大层级
    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
    return cur_c;  // 返回新添加的元素 ID
  }

  // 搜索 K 最近邻
  std::priority_queue<std::pair<float, labeltype>> searchKnn(const void* query_data, size_t k, uint64_t ef,
                                                             BaseFilterFunctor* isIdAllowed = nullptr) const override {
    std::priority_queue<std::pair<float, labeltype>> result;  // 存储结果的优先队列
    if (cur_element_count_ == 0) return result;               // 如果没有元素，直接返回空队列

    // 归一化查询数据
    std::shared_ptr<float[]> normalize_query;
    normalize_vector(query_data, normalize_query);

    // 从入口节点开始进行搜索
    tableint currObj = enterpoint_node_;
    float curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    // 从最大层级开始，逐层进行搜索
    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int* data;

        // 获取当前节点在该层的邻居列表
        data = (unsigned int*)get_linklist(currObj, level);
        int size = getListCount(data);
        metric_hops_++;                         // 统计跳数
        metric_distance_computations_ += size;  // 统计计算的距离次数

        tableint* datal = (tableint*)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");

          // 计算当前候选节点的距离
          float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

          // 如果距离更小，则更新当前节点和最小距离
          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    // 使用优先队列存储候选的元素，根据距离进行排序
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        top_candidates(allocator_);

    // 执行搜索，获取候选节点
    if (num_deleted_) {
      top_candidates = searchBaseLayerST<true, true>(currObj, query_data, std::max(ef, k), isIdAllowed);
    } else {
      top_candidates = searchBaseLayerST<false, true>(currObj, query_data, std::max(ef, k), isIdAllowed);
    }

    // 保留前 k 个最近邻
    while (top_candidates.size() > k) {
      top_candidates.pop();
    }

    // 将结果转换为带有标签的优先队列
    while (top_candidates.size() > 0) {
      std::pair<float, tableint> rez = top_candidates.top();
      result.push(std::pair<float, labeltype>(rez.first, getExternalLabel(rez.second)));
      top_candidates.pop();
    }

    return result;  // 返回最终的 K 最近邻结果
  }

  // 搜索范围内的邻居
  std::priority_queue<std::pair<float, labeltype>> searchRange(
      const void* query_data, float radius, uint64_t ef, BaseFilterFunctor* isIdAllowed = nullptr) const override {
    std::priority_queue<std::pair<float, labeltype>> result;  // 存储结果的优先队列
    if (cur_element_count_ == 0) return result;               // 如果没有元素，直接返回空队列

    // 归一化查询数据
    std::shared_ptr<float[]> normalize_query;
    normalize_vector(query_data, normalize_query);

    // 从入口节点开始进行搜索
    tableint currObj = enterpoint_node_;
    float curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    // 从最大层级开始，逐层进行搜索
    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int* data;

        // 获取当前节点在该层的邻居列表
        data = (unsigned int*)get_linklist(currObj, level);
        int size = getListCount(data);
        metric_hops_++;                         // 统计跳数
        metric_distance_computations_ += size;  // 统计计算的距离次数

        tableint* datal = (tableint*)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");

          // 计算当前候选节点的距离
          float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

          // 如果距离更小，则更新当前节点和最小距离
          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    // 使用优先队列存储候选的元素，根据距离进行排序
    std::priority_queue<std::pair<float, tableint>, vsag::Vector<std::pair<float, tableint>>, CompareByFirst>
        top_candidates(allocator_);

    // 如果有删除的元素，抛出异常
    if (num_deleted_) {
      throw std::runtime_error("not support perform range search on a index that deleted some vectors");
    } else {
      // 执行范围搜索，获取符合条件的候选节点
      top_candidates = searchBaseLayerST<false, true>(currObj, query_data, radius, ef, isIdAllowed);
    }

    // 将结果转换为带有标签的优先队列
    while (top_candidates.size() > 0) {
      std::pair<float, tableint> rez = top_candidates.top();
      result.push(std::pair<float, labeltype>(rez.first, getExternalLabel(rez.second)));
      top_candidates.pop();
    }

    return result;  // 返回最终的范围搜索结果
  }

  // 重置内部内存空间
  void reset() {
    // 释放每个分配的内存块，并将指针设置为 nullptr
    allocator_->Deallocate(element_levels_);
    element_levels_ = nullptr;
    allocator_->Deallocate(reversed_level0_link_list_);
    reversed_level0_link_list_ = nullptr;
    allocator_->Deallocate(reversed_link_lists_);
    reversed_link_lists_ = nullptr;
    allocator_->Deallocate(molds_);
    molds_ = nullptr;
    allocator_->Deallocate(link_lists_);
    link_lists_ = nullptr;
  }

  // 初始化内存空间
  bool init_memory_space() override {
    // 先重置内存空间
    reset();

    // 为每个元素分配内存用于存储层级信息
    element_levels_ = (int*)allocator_->Allocate(max_elements_ * sizeof(int));
    // 调整第0层数据的内存空间
    if (not data_level0_memory_->Resize(max_elements_)) {
      throw std::runtime_error("allocate data_level0_memory_ error");
    }

    // 如果使用反向边，则为反向链接分配内存
    if (use_reversed_edges_) {
      reversed_level0_link_list_ = (reverselinklist**)allocator_->Allocate(max_elements_ * sizeof(reverselinklist*));
      if (reversed_level0_link_list_ == nullptr) {
        throw std::runtime_error("allocate reversed_level0_link_list_ fail");
      }
      // 初始化反向链接列表为 0
      memset(reversed_level0_link_list_, 0, max_elements_ * sizeof(reverselinklist*));

      // 为反向链接哈希表分配内存
      reversed_link_lists_ = (vsag::UnorderedMap<int, reverselinklist>**)allocator_->Allocate(
          max_elements_ * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
      if (reversed_link_lists_ == nullptr) {
        throw std::runtime_error("allocate reversed_link_lists_ fail");
      }
      // 初始化反向链接哈希表为 0
      memset(reversed_link_lists_, 0, max_elements_ * sizeof(vsag::UnorderedMap<int, reverselinklist>*));
    }

    // 如果需要归一化数据，分配归一化数据的存储空间
    if (normalize_) {
      ip_func_ = vsag::InnerProduct;  // 设置内积计算方法
      molds_ = (float*)allocator_->Allocate(max_elements_ * sizeof(float));
    }

    // 为每个元素分配内存用于存储链接列表
    link_lists_ = (char**)allocator_->Allocate(sizeof(void*) * max_elements_);
    if (link_lists_ == nullptr)
      throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
    // 初始化链接列表为 0
    memset(link_lists_, 0, sizeof(void*) * max_elements_);

    return true;  // 成功初始化内存空间
  }

  // 检查数据结构的完整性
  void checkIntegrity() {
    int connections_checked = 0;                                                   // 用于统计检查的连接数
    vsag::Vector<int> inbound_connections_num(cur_element_count_, 0, allocator_);  // 存储每个元素的入度

    // 遍历所有元素和它们的层级
    for (int i = 0; i < cur_element_count_; i++) {
      // 遍历当前元素的所有层级
      for (int l = 0; l <= element_levels_[i]; l++) {
        // 获取当前层级的链接列表
        linklistsizeint* ll_cur = get_linklist_at_level(i, l);
        int size = getListCount(ll_cur);
        tableint* data = (tableint*)(ll_cur + 1);  // 获取链接数据

        // 使用哈希集来检测重复的链接
        vsag::UnorderedSet<tableint> s(allocator_);
        for (int j = 0; j < size; j++) {
          assert(data[j] > 0);                   // 确保链接有效
          assert(data[j] < cur_element_count_);  // 确保链接指向有效的元素
          assert(data[j] != i);                  // 确保没有自环
          inbound_connections_num[data[j]]++;    // 统计入度
          s.insert(data[j]);                     // 将链接加入哈希集
          connections_checked++;                 // 增加检查的连接数
        }
        assert(s.size() == size);  // 确保没有重复的链接
      }
    }

    // 如果元素数量大于 1，输出最小和最大入度
    if (cur_element_count_ > 1) {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      for (int i = 0; i < cur_element_count_; i++) {
        assert(inbound_connections_num[i] > 0);             // 确保每个元素的入度大于 0
        min1 = std::min(inbound_connections_num[i], min1);  // 更新最小入度
        max1 = std::max(inbound_connections_num[i], max1);  // 更新最大入度
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";  // 输出入度范围
    }

    // 输出完整性检查结果
    std::cout << "integrity ok, checked " << connections_checked << " connections\n";
  }

}  // namespace hnswlib