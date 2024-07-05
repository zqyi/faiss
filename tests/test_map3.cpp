/*
 *  Trade secret of Alibaba Group R&D.
 *  Copyright (c) 2010 Alibaba Group R&D. (unpublished)
 *
 *  All rights reserved.  This notice is intended as a precaution against
 *  inadvertent publication and does not imply publication or any waiver
 *  of confidentiality.  The year included in the foregoing notice is the
 *  year of creation of the work.
 *
 */
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap3.h"
#include <iostream>
#include <vector>
#include <cstdlib> // for drand48
#include <faiss/index_factory.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexIVF.h>
using namespace std;
int main() {
    // 设置向量的维度和数据数量
    int d = 128;      // 向量维度
    int nb = 10000;   // 数据库向量数量
    int nq = 10;      // 查询向量数量

    // 生成数据库向量和查询向量
    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[d * i + j] = drand48();
        }
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) {
            xq[d * i + j] = drand48();
        }
    }

    // 创建基础索
    faiss::Index* baseIndex= faiss::index_factory(d,
                        "IVF128,Flat" );
    faiss::IndexIVF* ivfIndex = dynamic_cast<faiss::IndexIVF*>(baseIndex);
    if (ivfIndex != nullptr) {
        ivfIndex->make_direct_map();
    } else {
        faiss::IndexBinaryIVF* binaryIvfIndex = dynamic_cast<faiss::IndexBinaryIVF*>(baseIndex);
        if(binaryIvfIndex != nullptr) {
            binaryIvfIndex->make_direct_map();
        }
    }

    // 使用IndexIDMap3包装基础索引
    faiss::IndexIDMap3 index_id_map(baseIndex);

    // 为数据库向量创建自定义ID
    std::vector<faiss::idx_t> ids(nb);
    for (int i = 0; i < nb; i++) {
        ids[i] = i % 100;  // 自定义ID，模拟重复ID的情况
    }

    // 添加向量和自定义ID到索引
    index_id_map.train(nb,xb.data());
    index_id_map.add_with_ids(nb, xb.data(), ids.data());

    // 创建直接映射

    // 搜索最近的5个邻居
    int k = 5;
    std::vector<faiss::idx_t> I(nq * k); // 最近邻ID
    std::vector<float> D(nq * k);        // 距离

    index_id_map.search(nq, xq.data(), k, D.data(), I.data());

    // 输出搜索结果
    std::cout << "搜索结果: " << std::endl;
    for (int i = 0; i < nq; i++) {
        std::cout << "查询 " << i << " : ";
        for (int j = 0; j < k; j++) {
            std::cout << "(" << I[i * k + j] << ", " << D[i * k + j] << ") ";
        }
        std::cout << std::endl;
    }

    // 获取具有特定ID的所有向量索引
    faiss::idx_t target_id = 42;
    std::vector<long> indices;
    // cout<<"idsize:"<<index_id_map.rev_map.size()<<endl;
    // for(auto itr = index_id_map.rev_map.begin();itr != index_id_map.rev_map.end();++itr)
    // {
    //     cout<<itr->first<<":"<<itr->second<<endl;
    // }
    // cout<<"idmapsize:"<<index_id_map.id_map.size()<<endl;
    // for(auto itr = index_id_map.id_map.begin();itr != index_id_map.id_map.end();++itr)
    // {
    //     cout<<*itr<<endl;
    // }

    for (long i = 0; i < (int)index_id_map.id_map.size(); ++i) {
        if (index_id_map.id_map.at(i) == target_id) {
            indices.push_back(i);
        }
    }

    // 重建具有特定ID的向量
    std::cout << "重建向量，ID为 " << target_id << "：" << std::endl;
   // for (long idx : indices) 
    {
        auto idx = target_id;
        // std::vector<float> vec(d);
        // index_id_map.reconstruct(idx, vec.data());

        float *vec = nullptr;
        size_t n = 0;

        index_id_map.reconstruct_multi(idx, vec, n);
        std::cout << "向量 " << idx << " :" << " chunking num "  << n << std::endl;
        // for(int i=0; i < n;i++){
        //     for (int j = 0; j < d; j++) {
        //         std::cout << " " << vec[i*d + j];
        //     }
        //     std::cout << std::endl;
        // }
        std::cout << std::endl;
    }
    cout<<"vector size:"<<index_id_map.ntotal<<endl;

    return 0;
}

