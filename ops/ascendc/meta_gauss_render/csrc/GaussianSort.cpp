/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file GaussianSort.cpp
 * \brief gaussian sort pybind adapter
 */

#include <string>

#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

at::Tensor gaussian_sort(const at::Tensor& lb_sched, const at::Tensor& gaussian_cnt, const at::Tensor& depths,
                         const at::Tensor& gs_ids, const at::Tensor& sorted_offset, int32_t max_tile_gauss)
{
    TORCH_CHECK(depths.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(depths.device() == lb_sched.device(), "Inconsistent device.");
    TORCH_CHECK(depths.sizes() == gs_ids.sizes(), "Invalid shape.");
    TORCH_CHECK(depths.scalar_type() == at::kFloat,
                "depths: float32 tensor expected but got a tensor with dtype: ", depths.scalar_type());

    auto device = depths.device();
    auto options = at::TensorOptions().dtype(at::kInt).layout(at::kStrided).device(device);
    int64_t sorted_total_nums = sorted_offset.index({-1}).item<int64_t>();
    at::Tensor sorted_gs_ids = at::empty({sorted_total_nums}, options);
    EXEC_NPU_CMD(aclnnGaussianSort, lb_sched, gaussian_cnt, depths, gs_ids, sorted_offset, max_tile_gauss,
                 sorted_gs_ids);

    return sorted_gs_ids;
}
