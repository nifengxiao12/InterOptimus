from setuptools import setup, find_packages

setup(
    name="InterOptimus",                # 包的名称
    version="1.0.0",                         # 版本号
    author="Yaoshu Xie",                      # 作者信息
    author_email="jasonxie.sz.tsinghua.edu.cn",   # 作者邮箱
    description="High througput simulation making crystalline interfaces",   # 简短描述
    long_description=open("README.md").read(),  # 从README中读取长描述
    long_description_content_type="text/markdown",  # 描述文件格式
    url="https://github.com/HouGroup/InterOptimus/",  # 项目主页
    packages=find_packages(),                # 自动查找项目中的包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 或其他许可证
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",                 # Python版本要求
    install_requires=[
        # 在这里列出项目依赖，比如：
        "requests",
        "numpy",
        "interface_master"
    ],
    entry_points={
        "console_scripts": [
            "your_command=your_package.module:function_name",  # 入口点
        ],
    },
)
