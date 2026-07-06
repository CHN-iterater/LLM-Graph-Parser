from setuptools import setup, find_packages

setup(
    name="llm-graph-parser",
    version="0.1.0",
    description="LLM Graph Parser - 大语言模型计算图的层级化解析工具",
    packages=find_packages(include=["llm_graph_parser", "llm_graph_parser.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24",
    ],
)
