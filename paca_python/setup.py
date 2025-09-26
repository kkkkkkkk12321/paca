"""
PACA v5 Python Edition Setup Script
Personal Adaptive Cognitive Assistant v5 설치 스크립트
"""

from setuptools import setup, find_packages
from pathlib import Path

# 프로젝트 루트 디렉토리
HERE = Path(__file__).parent

# README 파일 읽기
README = (HERE / "README.md").read_text(encoding="utf-8")

# 요구사항 파일 읽기
REQUIREMENTS = (HERE / "requirements.txt").read_text(encoding="utf-8").strip().split('\n')

setup(
    name="paca-python",
    version="5.0.0",
    author="PACA Development Team",
    author_email="paca-team@example.com",
    description="Personal Adaptive Cognitive Assistant v5 - Python Edition",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/paca-team/paca-python",
    project_urls={
        "Bug Reports": "https://github.com/paca-team/paca-python/issues",
        "Source": "https://github.com/paca-team/paca-python",
        "Documentation": "https://paca-python.readthedocs.io",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.0.0",
            "pre-commit>=3.0.0",
        ],
        "gui": [
            "customtkinter>=5.0.0",
            "Pillow>=9.0.0",
            "pystray>=0.19.0",
        ],
        "nlp": [
            "konlpy>=0.6.0",
            "transformers>=4.20.0",
            "torch>=1.13.0",
            "sentencepiece>=0.1.99",
        ],
        "all": [
            "customtkinter>=5.0.0",
            "Pillow>=9.0.0",
            "pystray>=0.19.0",
            "konlpy>=0.6.0",
            "transformers>=4.20.0",
            "torch>=1.13.0",
            "sentencepiece>=0.1.99",
        ]
    },
    entry_points={
        "console_scripts": [
            "paca=paca.__main__:main",
            "paca-desktop=desktop_app.main:main",
            "paca-optimize=scripts.performance_optimizer:main",
            "paca-package=scripts.setup_packaging:main",
        ],
    },
    include_package_data=True,
    package_data={
        "paca": ["py.typed"],
        "desktop_app": ["assets/**/*"],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "cognitive assistant",
        "ACT-R",
        "SOAR",
        "korean nlp",
        "chatbot",
        "reasoning",
        "machine learning"
    ],
)