"""
GPU诊断脚本 - 检查PyTorch和CUDA是否正常工作
"""
import torch
import sys

print("=" * 60)
print("PyTorch和CUDA诊断")
print("=" * 60)

# 1. 检查Python版本
print(f"\n1. Python版本: {sys.version}")

# 2. 检查PyTorch版本
print(f"\n2. PyTorch版本: {torch.__version__}")

# 3. 检查CUDA是否可用
print(f"\n3. CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 4. 检查CUDA版本
    print(f"\n4. CUDA版本: {torch.version.cuda}")

    # 5. 检查cuDNN版本
    print(f"\n5. cuDNN版本: {torch.backends.cudnn.version()}")

    # 6. 检查GPU信息
    print(f"\n6. GPU信息:")
    print(f"   - GPU数量: {torch.cuda.device_count()}")
    print(f"   - GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"   - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 7. 测试GPU基本操作
    print(f"\n7. 测试GPU基本操作:")
    try:
        # 清理缓存
        torch.cuda.empty_cache()
        print(f"   - 清理GPU缓存: 成功")

        # 创建小张量
        x = torch.randn(10, 10).cuda()
        print(f"   - 创建小张量: 成功")

        # 矩阵乘法
        y = torch.matmul(x, x)
        print(f"   - 矩阵乘法: 成功")

        # 清理
        del x, y
        torch.cuda.empty_cache()
        print(f"   - 清理张量: 成功")

        # 测试更大的张量
        print(f"\n8. 测试更大的张量:")
        sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
        for size in sizes:
            try:
                torch.cuda.empty_cache()
                x = torch.randn(*size).cuda()
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                print(f"   - {size}: 成功")
            except Exception as e:
                print(f"   - {size}: 失败 - {e}")
                break

        print(f"\n✓ GPU基本功能正常!")

    except Exception as e:
        print(f"\n✗ GPU测试失败: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"\n✗ CUDA不可用，请检查:")
    print("   1. 是否安装了CUDA版本的PyTorch")
    print("   2. GPU驱动是否正确安装")
    print("   3. GPU是否被系统识别")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
