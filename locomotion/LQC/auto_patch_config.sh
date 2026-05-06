#!/bin/bash

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
CONFIG_FILE="${BASE_DIR}/algorithms/utils/config.py"

# 检查文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误：未找到 config.py 文件！"
    echo "请确保你在 LQC 目录下执行此脚本！"
    echo "当前目录：$BASE_DIR"
    echo "目标文件：$CONFIG_FILE"
    exit 1
fi

echo "✅ 找到文件：$CONFIG_FILE"
echo "✅ 精准匹配指定 print + raise 两行，在这两行之后紧接着插入所需代码..."

# 精准匹配 print + raise 两行之后进行插入
sed -i '/^[[:space:]]*print(f"Failed to initialize DDP process group: {e}")/{
n
n
a\
        if self.ddp:\
            cpu_count = os.cpu_count()\
            omp_threads = max(1, cpu_count // self.world_size)\
            os.environ["OMP_NUM_THREADS"] = str(omp_threads)\
            core_start = self.local_rank * omp_threads\
            core_end = core_start + omp_threads - 1\
            os.environ["GOMP_CPU_AFFINITY"] = f"{core_start}-{core_end}"\
            print(f"DDP: set OMP_NUM_THREADS={omp_threads}, GOMP_CPU_AFFINITY={core_start}-{core_end} "\
                  f"(cpu_count={cpu_count}, world_size={self.world_size})")
}' "$CONFIG_FILE"

# ===================== 【校验是否真的插入成功】 =====================
if grep -q 'GOMP_CPU_AFFINITY' "$CONFIG_FILE"; then
    echo ""
    echo "✅ 插入成功！位置正确！"
    echo "✅ 已自动设置 OMP 线程与 CPU 亲和性！"
else
    echo ""
    echo "❌ 错误：未找到目标位置，插入失败！"
    echo "请检查 config.py 中是否存在以下两行："
    echo "   print(f\"Failed to initialize DDP process group: {e}\")"
    echo "   raise"
    exit 1
fi