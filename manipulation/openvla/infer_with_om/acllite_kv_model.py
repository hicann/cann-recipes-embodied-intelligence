# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (R) @huawei.com; all rights reserved
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
Extended AclLiteModel for KV Cache Optimization (Device-side update).
Based on AclLiteModel from Ascend samples.
"""
import os
import time

import acl
import acllite_utils as utils
import constants as const
import numpy as np
from acllite_logger import log_error, log_info, log_warning
from acllite_resource import resource_list

DTYPE_MAPPING = {
    const.ACL_FLOAT: np.float32,
    const.ACL_FLOAT16: np.float16,
    const.ACL_INT64: np.int64,
    const.ACL_INT32: np.int32,
    const.ACL_BOOL: np.bool_,
}


class KVStatefulModel:
    """
    Wrap ACL model inference interface with support for device-resident memory (KV cache).
    """

    def __init__(self, model_path, load_type=0):
        self._run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

        self._copy_policy = const.ACL_MEMCPY_DEVICE_TO_DEVICE
        if self._run_mode == const.ACL_HOST:
            self._copy_policy = const.ACL_MEMCPY_DEVICE_TO_HOST

        self._model_path = model_path
        self._load_type = load_type
        self._model_id = None
        self._input_num = 0
        self._input_buffer = []
        self._input_dataset = None
        self._output_dataset = None
        self._model_desc = None
        self._output_size = 0
        self.input_buffers = []        # Device memory pointers for inputs
        self.input_buffer_sizes = []   # Sizes of input buffers
        ret = self._init_resource()
        utils.check_ret("KVStatefulModel._init_resource", ret)
        self._is_destroyed = False
        resource_list.register(self)

    @staticmethod
    def _acl_dtype_to_numpy(datatype):
        return DTYPE_MAPPING.get(datatype, np.float32)

    def _init_resource(self):
        log_info("Init model resource start...")
        if not os.path.isfile(self._model_path):
            log_error(f"model_path is not a file, please check. model_path={self._model_path}")
            return const.FAILED

        if self._load_type == 0:
            self._model_id, ret = acl.mdl.load_from_file(self._model_path)
            utils.check_ret("acl.mdl.load_from_file", ret)
        elif self._load_type == 1:
            with open(self._model_path, "rb") as f:
                om_bytes = f.read()
            if om_bytes:
                ptr = acl.util.bytes_to_ptr(om_bytes)
                self._model_id, ret = acl.mdl.load_from_mem(ptr, len(om_bytes))
                utils.check_ret("acl.mdl.load_from_mem", ret)
            else:
                log_error(
                    "Failed to read model file or model file is empty "
                    f"(om_bytes is empty), please check. model_path={self._model_path}"
                )
                return const.FAILED
        
        self._model_desc = acl.mdl.create_desc()
        if not self._model_desc:
            log_error("acl.mdl.create_desc failed")
            if self._model_id:
                acl.mdl.unload(self._model_id)
                self._model_id = None
            utils.check_ret("acl.mdl.create_desc", const.FAILED)
        ret = acl.mdl.get_desc(self._model_desc, self._model_id)
        utils.check_ret("acl.mdl.get_desc", ret)
        
        self._output_size = acl.mdl.get_num_outputs(self._model_desc)
        # Create output dataset
        self._gen_output_dataset(self._output_size)
        
        # Init input buffers (allocate device memory once)
        self._input_num = acl.mdl.get_num_inputs(self._model_desc)
        self._init_input_buffer() # But we still need buffers for numpy inputs
        
        log_info("Init model resource success")
        return const.SUCCESS

    def _init_input_buffer(self):
        # Pre-allocate lists with known size to avoid dynamic resizing
        self.input_buffers = [None] * self._input_num
        self.input_buffer_sizes = [0] * self._input_num
        for i in range(self._input_num):
            input_buffer_size = acl.mdl.get_input_size_by_index(self._model_desc, i)
            # Malloc device memory for input (used as fallback/cache for numpy inputs)
            input_buffer, ret = acl.rt.malloc(input_buffer_size, const.ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != const.ACL_SUCCESS:
                log_error(
                    f"acl.rt.malloc failed when creating input buffer[{i}], size={input_buffer_size}, ret={ret}"
                )
                for j in range(i):
                    if self.input_buffers[j]:
                        acl.rt.free(self.input_buffers[j])
                        self.input_buffers[j] = None
                        self.input_buffer_sizes[j] = 0
                utils.check_ret("acl.rt.malloc", ret)
            self.input_buffers[i] = input_buffer
            self.input_buffer_sizes[i] = input_buffer_size

    def _gen_output_dataset(self, output_num):
        log_info("KVStatefulModel create model output dataset:")
        dataset = acl.mdl.create_dataset()
        if not dataset:
            log_error("acl.mdl.create_dataset failed when creating output dataset")
            utils.check_ret("acl.mdl.create_dataset", const.FAILED)
        for i in range(output_num):
            size = acl.mdl.get_output_size_by_index(self._model_desc, i)
            # Malloc device memory for output
            buf, ret = acl.rt.malloc(size, const.ACL_MEM_MALLOC_NORMAL_ONLY)
            if ret != const.ACL_SUCCESS:
                log_error(f"acl.rt.malloc failed when creating output buffer[{i}], size={size}, ret={ret}")
                self._destroy_dataset_with_buffers(dataset)
                utils.check_ret("acl.rt.malloc", ret)
            
            dataset_buffer = acl.create_data_buffer(buf, size)
            if not dataset_buffer:
                log_error(f"acl.create_data_buffer failed for output {i}, size={size}")
                acl.rt.free(buf)
                self._destroy_dataset_with_buffers(dataset)
                utils.check_ret("acl.create_data_buffer", const.FAILED)
            _, ret = acl.mdl.add_dataset_buffer(dataset, dataset_buffer)
            if ret:
                log_error(f"acl.mdl.add_dataset_buffer failed for output {i}, ret={ret}")
                acl.destroy_data_buffer(dataset_buffer)
                acl.rt.free(buf)
                self._destroy_dataset_with_buffers(dataset)
                utils.check_ret("acl.mdl.add_dataset_buffer", ret)
        self._output_dataset = dataset
        log_info("Create model output dataset success")

    def _gen_input_dataset(self):
        self._input_dataset = acl.mdl.create_dataset()
        if not self._input_dataset:
            log_error("acl.mdl.create_dataset failed when creating input dataset")
            utils.check_ret("acl.mdl.create_dataset", const.FAILED)
        # Pre-allocate lists with known size to avoid dynamic resizing
        self.input_buffers = [None] * self._input_num
        self.input_buffer_sizes = [0] * self._input_num
        for i in range(self._input_num):
            input_buffer_size = acl.mdl.get_input_size_by_index(self._model_desc, i)
            # Malloc device memory for input
            input_buffer, ret = acl.rt.malloc(input_buffer_size, const.ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != const.ACL_SUCCESS:
                log_error(
                    f"acl.rt.malloc failed when creating input dataset buffer[{i}], size={input_buffer_size}, ret={ret}"
                )
                self._destroy_dataset_with_buffers(self._input_dataset)
                self._input_dataset = None
                utils.check_ret("acl.rt.malloc", ret)
            
            input_data = acl.create_data_buffer(input_buffer, input_buffer_size)
            if not input_data:
                log_error(f"acl.create_data_buffer failed for input {i}, size={input_buffer_size}")
                acl.rt.free(input_buffer)
                self._destroy_dataset_with_buffers(self._input_dataset)
                self._input_dataset = None
                utils.check_ret("acl.create_data_buffer", const.FAILED)
            _, ret = acl.mdl.add_dataset_buffer(self._input_dataset, input_data)
            if ret != const.ACL_SUCCESS:
                log_error(f"acl.mdl.add_dataset_buffer failed for input {i}, ret={ret}")
                acl.destroy_data_buffer(input_data)
                acl.rt.free(input_buffer)
                self._destroy_dataset_with_buffers(self._input_dataset)
                self._input_dataset = None
                utils.check_ret("acl.mdl.add_dataset_buffer", ret)
            
            self.input_buffers[i] = input_buffer
            self.input_buffer_sizes[i] = input_buffer_size

    def execute(self, input_list, output_to_host=True):
        """
        Execute inference.
        
        Args:
            input_list: List of inputs. Each element can be:
                - np.ndarray: Will be copied to device (H2D) using internal buffer.
                - {'ptr': int, 'size': int}: Device pointer (Zero Copy).
            output_to_host: If True, copy outputs to Host (numpy). 
                            If False, return list of {'ptr': int, 'size': int} on Device.
        """
        if self._run_mode == const.ACL_DEVICE:
            h2d_kind = const.ACL_MEMCPY_DEVICE_TO_DEVICE
        else:
            h2d_kind = const.ACL_MEMCPY_HOST_TO_DEVICE

        # Create dynamic dataset for Zero Copy support
        dataset = acl.mdl.create_dataset()
        if not dataset:
            log_error("acl.mdl.create_dataset failed in execute")
            utils.check_ret("acl.mdl.create_dataset", const.FAILED)
        data_buffers = [] # Keep track to destroy later

        # Input Copy / Wrap
        copy_start = time.time()
        for i, data_in in enumerate(input_list):
            target_buffer = self.input_buffers[i]
            target_size = self.input_buffer_sizes[i]
            
            # Case 1: Device Pointer (Zero Copy)
            if isinstance(data_in, dict) and 'ptr' in data_in and 'size' in data_in:
                src_ptr = data_in['ptr']
                src_size = data_in['size']
                
                if src_size != target_size:
                    log_warning(f"Input[{i}] Zero Copy size mismatch: src={src_size}, dst={target_size}")
                
                # Use the pointer directly! No memcpy.
                ptr = src_ptr
                size = src_size

            # Case 2: Numpy Array (Host data)
            elif isinstance(data_in, np.ndarray):
                src_size = data_in.size * data_in.itemsize
                if src_size != target_size:
                    log_warning(f"Input[{i}] H2D size mismatch: src={src_size}, dst={target_size}")
                
                if "bytes_to_ptr" in dir(acl.util):
                    bytes_data = data_in.tobytes()
                    host_ptr = acl.util.bytes_to_ptr(bytes_data)
                else:
                    host_ptr = acl.util.numpy_to_ptr(data_in)
                
                # Copy to internal fixed buffer
                ret = acl.rt.memcpy(target_buffer, target_size,
                                    host_ptr, src_size,
                                    h2d_kind)
                utils.check_ret("acl.rt.memcpy H2D", ret)
                
                ptr = target_buffer
                size = target_size
            
            else:
                log_error(f"Unsupported input type at index {i}: {type(data_in)}")
                for db in data_buffers:
                    acl.destroy_data_buffer(db)
                acl.mdl.destroy_dataset(dataset)
                return None
            
            # Add to dataset
            db = acl.create_data_buffer(ptr, size)
            if not db:
                log_error(f"acl.create_data_buffer failed for input {i}, size={size}")
                for old_db in data_buffers:
                    acl.destroy_data_buffer(old_db)
                acl.mdl.destroy_dataset(dataset)
                utils.check_ret("acl.create_data_buffer", const.FAILED)
            _, ret = acl.mdl.add_dataset_buffer(dataset, db)
            if ret != const.ACL_SUCCESS:
                log_error(f"add_dataset_buffer failed for input {i}, ret={ret}")
                acl.destroy_data_buffer(db)
                for old_db in data_buffers:
                    acl.destroy_data_buffer(old_db)
                acl.mdl.destroy_dataset(dataset)
                utils.check_ret("acl.mdl.add_dataset_buffer", ret)
            data_buffers.append(db)

        copy_end = time.time()
        log_info(
            f"[KVStatefulModel] Input prepare time: {(copy_end - copy_start) * 1000:.2f} ms"
        )

        # Execute
        start_time = time.time()
        ret = acl.mdl.execute(self._model_id, dataset, self._output_dataset)
        end_time = time.time()
        log_info(
            f"[KVStatefulModel] Model execute time: {(end_time - start_time) * 1000:.2f} ms"
        )
        if ret != const.ACL_SUCCESS:
            for db in data_buffers:
                acl.destroy_data_buffer(db)
            acl.mdl.destroy_dataset(dataset)
            utils.check_ret("acl.mdl.execute", ret)

        # Cleanup input dataset wrappers (but NOT the memory if it was passed in)
        for db in data_buffers:
            acl.destroy_data_buffer(db)
        acl.mdl.destroy_dataset(dataset)

        if output_to_host:
            return self._output_dataset_to_numpy()
        else:
            return self._output_dataset_to_ptr()

    def _output_dataset_to_numpy(self):
        dataset = []
        num = acl.mdl.get_dataset_num_buffers(self._output_dataset)
        
        # We need to know output shapes/types to reconstruct numpy arrays
        # This is expensive to query every time, but sticking to AclLite logic for now
        output_meta = self._gen_output_tensor_meta()

        for i in range(num):
            buf = acl.mdl.get_dataset_buffer(self._output_dataset, i)
            dev_ptr = acl.get_data_buffer_addr(buf)
            dev_size = int(acl.get_data_buffer_size(buf))
            
            # Prepare host buffer
            host_buffer = output_meta[i]["buffer"] # pre-allocated numpy
            
            # Copy D2H
            # Using raw pointer approach
            if "ptr" in output_meta[i]:
                dst_ptr = output_meta[i]["ptr"]
                ret = acl.rt.memcpy(dst_ptr, dev_size, dev_ptr, dev_size, self._copy_policy)
                utils.check_ret("acl.rt.memcpy D2H", ret)
                
                # Convert bytes to numpy if needed
                if isinstance(host_buffer, bytes):
                    # Reconstruct from bytes
                    # Note: AclLiteModel logic is a bit complex here depending on implementation
                    # Simplified: assume we got bytes, need to view as numpy
                    np_array = np.frombuffer(
                        host_buffer, dtype=output_meta[i]["dtype"]
                    ).reshape(output_meta[i]["shape"])
                    dataset.append(np_array.copy())
                else:
                    dataset.append(host_buffer)
            else:
                # Fallback if gen_output_tensor_meta behaves differently
                pass
                
        return dataset

    def _gen_output_tensor_meta(self):
        # Helper to pre-allocate host buffers for D2H copy
        # Pre-allocate list with known size to avoid dynamic resizing
        meta_list = [None] * self._output_size
        for i in range(self._output_size):
            dims = acl.mdl.get_output_dims(self._model_desc, i)
            shape = tuple(dims[0]["dims"])
            datatype = acl.mdl.get_output_data_type(self._model_desc, i)
            size = acl.mdl.get_output_size_by_index(self._model_desc, i)
            
            np_type = self._acl_dtype_to_numpy(datatype)
            
            # Allocate numpy array
            # Calculate element count from size to be safe (size is bytes)
            elem_size = np.dtype(np_type).itemsize
            elem_count = size // elem_size
            
            # Ensure shape matches size
            # If dynamic shape, shape might be 0 or -1, use flat buffer
            # Here assuming static output shape for simplicity or flattening
            
            host_arr = np.zeros(elem_count, dtype=np_type)
            
            # Get pointer
            if "bytes_to_ptr" in dir(acl.util):
                bytes_data = host_arr.tobytes()
                ptr = acl.util.bytes_to_ptr(bytes_data)
                meta_list[i] = {"ptr": ptr, "buffer": bytes_data, "dtype": np_type, "shape": shape}
            else:
                ptr = acl.util.numpy_to_ptr(host_arr)
                meta_list[i] = {"ptr": ptr, "buffer": host_arr, "dtype": np_type, "shape": shape}
                
        return meta_list

    def _output_dataset_to_ptr(self):
        """
        Return list of device pointers and sizes. No copy.
        """
        ptrs = []
        num = acl.mdl.get_dataset_num_buffers(self._output_dataset)
        for i in range(num):
            buf = acl.mdl.get_dataset_buffer(self._output_dataset, i)
            dev_ptr = acl.get_data_buffer_addr(buf)
            dev_size = int(acl.get_data_buffer_size(buf))
            ptrs.append({'ptr': dev_ptr, 'size': dev_size})
        return ptrs

    def copy_output_to_host(self, index, ptr_info):
        """
        Copy specific output index to host numpy array.
        Args:
            index: Output index
            ptr_info: {'ptr': int, 'size': int} from execute result
        """
        dims = acl.mdl.get_output_dims(self._model_desc, index)
        shape = tuple(dims[0]["dims"])
        datatype = acl.mdl.get_output_data_type(self._model_desc, index)
        
        np_type = self._acl_dtype_to_numpy(datatype)
        
        # Prepare host buffer
        elem_size = np.dtype(np_type).itemsize
        elem_count = ptr_info['size'] // elem_size
        host_arr = np.zeros(elem_count, dtype=np_type)
        
        if "bytes_to_ptr" in dir(acl.util):
            bytes_data = host_arr.tobytes()
            dst_ptr = acl.util.bytes_to_ptr(bytes_data)
            ret = acl.rt.memcpy(dst_ptr, ptr_info['size'], ptr_info['ptr'], ptr_info['size'], self._copy_policy)
            utils.check_ret("acl.rt.memcpy specific D2H", ret)
            # frombuffer returns a view on bytes_data; copy() detaches and makes result writable.
            return np.frombuffer(bytes_data, dtype=np_type).reshape(shape).copy()
        else:
            dst_ptr = acl.util.numpy_to_ptr(host_arr)
            ret = acl.rt.memcpy(dst_ptr, ptr_info['size'], ptr_info['ptr'], ptr_info['size'], self._copy_policy)
            utils.check_ret("acl.rt.memcpy specific D2H", ret)
            return host_arr.reshape(shape).copy()

    @staticmethod
    def _destroy_dataset_with_buffers(dataset):
        if not dataset:
            return
        for i in range(acl.mdl.get_dataset_num_buffers(dataset)):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf:
                addr = acl.get_data_buffer_addr(data_buf)
                acl.destroy_data_buffer(data_buf)
                if addr:
                    acl.rt.free(addr)
        acl.mdl.destroy_dataset(dataset)

    def destroy(self):
        if self._is_destroyed:
            return
        # Release inputs
        if self._input_dataset:
            self._destroy_dataset_with_buffers(self._input_dataset)
            self._input_dataset = None
        
        # Release outputs
        if self._output_dataset:
            self._destroy_dataset_with_buffers(self._output_dataset)
            self._output_dataset = None

        if self._model_id:
            acl.mdl.unload(self._model_id)
        if self._model_desc:
            acl.mdl.destroy_desc(self._model_desc)
            
        self._is_destroyed = True
        resource_list.unregister(self)
        log_info("KVStatefulModel release source success")

    def __del__(self):
        self.destroy()
