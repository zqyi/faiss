#pragma once

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/IndexIDMap.h>

#include <unordered_map>
#include <vector>
#include "faiss/MetricType.h"

namespace faiss {

template <typename IndexT>
struct IndexIDMap3Template : IndexIDMapTemplate<IndexT> {
    using component_t = typename IndexT::component_t;
    using distance_t = typename IndexT::distance_t;

    std::unordered_map<idx_t, std::vector<idx_t>> rev_map;

    explicit IndexIDMap3Template(IndexT* index);

    /// make the rev_map from scratch
    void construct_rev_map();

    void add_with_ids(idx_t n, const component_t* x, const idx_t* xids)
            override;

    size_t remove_ids(const IDSelector& sel) override;

    void reconstruct(idx_t key, component_t* recons) const override;

    void reconstruct_multi(idx_t key, component_t*& recons, size_t &n) const;

    /// check that the rev_map and the id_map are in sync
    void check_consistency() const;

    void merge_from(IndexT& otherIndex, idx_t add_id = 0) override;

    ~IndexIDMap3Template() override {}
    IndexIDMap3Template() {}
};

using IndexIDMap3 = IndexIDMap3Template<Index>;
using IndexBinaryIDMap3 = IndexIDMap3Template<IndexBinary>;

} // namespace faiss