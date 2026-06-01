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
 * \file gaussian_sort_tiling.h
 * \brief gaussian sort op host tiling
 */

#ifndef GAUSSIAN_SORT_TILING_H
#define GAUSSIAN_SORT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GaussianSortTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, cameraNum);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, gaussNum);
TILING_DATA_FIELD_DEF(uint32_t, scheduleNum);
TILING_DATA_FIELD_DEF(uint32_t, maxSortNum);      // UB单次最大支持排序高斯球数
TILING_DATA_FIELD_DEF(uint32_t, maxMaskNum);      // 最大高斯球相交数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GaussianSort, GaussianSortTilingData)
}  // namespace optiling

#endif