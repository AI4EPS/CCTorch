from setuptools import setup

setup(
    name="CCTorch",
    version="0.1.0",
    long_description="Cross-Correlation using Pytorch",
    long_description_content_type="text/markdown",
    packages=["cctorch"],
    install_requires=["torch", "torch-vision", "h5py", "matplotlib", "pandas"],
)
