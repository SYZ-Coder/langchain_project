# test_imports.py
import sys
import subprocess
import pkg_resources

def check_packages():
    print("=== 包安装状态检查 ===")
    print(f"Python 解释器: {sys.executable}")
    print(f"虚拟环境: {sys.prefix}")
    print()

    # 检查关键包
    packages = [
        'langchain-openai',
        'langchain',
        'langserve',
        'fastapi',
        'uvicorn',
        'openai'
    ]

    for package in packages:
        try:
            # 检查包是否安装
            dist = pkg_resources.get_distribution(package)
            print(f"✅ {package} == {dist.version}")

            # 尝试导入
            if package == 'langchain-openai':
                import langchain_openai
                print(f"   📦 langchain_openai 导入成功")
            elif package == 'langserve':
                import langserve
                print(f"   📦 langserve 导入成功")
            else:
                __import__(package.replace('-', '_'))
                print(f"   📦 {package} 导入成功")

        except pkg_resources.DistributionNotFound:
            print(f"❌ {package} - 未安装")
        except ImportError as e:
            print(f"⚠️  {package} - 已安装但导入失败: {e}")
        except Exception as e:
            print(f"❓ {package} - 检查出错: {e}")

if __name__ == "__main__":
    check_packages()
