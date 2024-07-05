#include <faiss/IndexIDMap3.h>

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <stdexcept>
#include <vector>
#include <string.h>

#include "faiss/MetricType.h"
#include "faiss/impl/FaissAssert.h"

namespace faiss{
/*****************************************************
 * IndexIDMap3 implementation
 *******************************************************/

template <typename IndexT>
IndexIDMap3Template<IndexT>::IndexIDMap3Template(IndexT* index)
        : IndexIDMapTemplate<IndexT>(index) {}


template <typename IndexT>
void IndexIDMap3Template<IndexT>::add_with_ids(
        idx_t n,
        const typename IndexT::component_t* x,
        const idx_t* xids) {
    size_t prev_ntotal = this->ntotal;    
    IndexIDMapTemplate<IndexT>::add_with_ids(n, x, xids);
    for (size_t i = prev_ntotal; i < this->ntotal; i++) {
        rev_map[this->id_map[i]].push_back(i);
    }
}

template <typename IndexT>
void IndexIDMap3Template<IndexT>::check_consistency() const {
    // check length
    size_t rev_map_size = 0;
    for (const auto& pair : rev_map) {
        rev_map_size += pair.second.size();
    }
    FAISS_THROW_IF_NOT(rev_map_size == this->ntotal);
    FAISS_THROW_IF_NOT(rev_map_size == this->id_map.size());

    // check value
    for (const auto& [doc_id, inner_id_vec] : rev_map){
        for (const idx_t& inner_id : inner_id_vec){
            FAISS_THROW_IF_NOT(this->id_map[inner_id] == doc_id);
        }
    }
}

template <typename IndexT>
void IndexIDMap3Template<IndexT>::merge_from(IndexT& otherIndex, idx_t add_id) {
    size_t prev_ntotal = this->ntotal;
    IndexIDMapTemplate<IndexT>::merge_from(otherIndex, add_id);

    for (size_t i = prev_ntotal; i < this->ntotal; i++) {
        rev_map[this->id_map[i]].push_back(i);
    }
    static_cast<IndexIDMap3Template<IndexT>&>(otherIndex).rev_map.clear();
}

template <typename IndexT>
void IndexIDMap3Template<IndexT>::construct_rev_map() {
    rev_map.clear();
    for (size_t i = 0; i < this->ntotal; i++) {
        rev_map[this->id_map[i]].push_back(i);
    }
}

template <typename IndexT>
size_t IndexIDMap3Template<IndexT>::remove_ids(const IDSelector& sel) {
    // This is quite inefficient
    size_t nremove = IndexIDMapTemplate<IndexT>::remove_ids(sel);
    construct_rev_map();
    return nremove;
}

template <typename IndexT>
void IndexIDMap3Template<IndexT>::reconstruct(
        idx_t key,
        typename IndexT::component_t* recons) const {
    try {
        // TODO: return multi recons
        auto &inner_id_vec = rev_map.at(key);

        FAISS_ASSERT(1 == inner_id_vec.size()); //FIXME: multi doc_id not implemented
        

        this->index->reconstruct(inner_id_vec[0], recons);
    } catch (const std::out_of_range&) {
        FAISS_THROW_FMT("key %" PRId64 " not found", key);
    }
}


template <typename IndexT>
void IndexIDMap3Template<IndexT>::reconstruct_multi(
    idx_t key,
    component_t*& recons, size_t& n) const {
    try {
        const auto& inner_ids = rev_map.at(key);

        size_t inner_n = inner_ids.size();
        if (inner_n == 0) {
            FAISS_THROW_FMT("key %" PRId64 " not found", key);
        }
        
        if (recons == nullptr){
            recons = new component_t[inner_n*this->index->d];
            std::fill_n(recons, inner_n*this->index->d, static_cast<component_t>(0));
        }else{
            FAISS_THROW_MSG("reconstruct_multi need alloc in faiss");
        }

        for(size_t i=0;i<inner_n;i++){
            this->index->reconstruct(inner_ids[i], &recons[i*(this->index->d)]);
        }

        n = inner_n;
    } catch (const std::out_of_range&) {
        FAISS_THROW_FMT("key %" PRId64 " not found", key);
    }
}

template struct IndexIDMap3Template<Index>;
template struct IndexIDMap3Template<IndexBinary>;

}// namespace faiss