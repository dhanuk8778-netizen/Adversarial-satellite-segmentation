from setuptools import find_packages, setup

setup(
    name="adversarial-satellite-segmentation",
    version="0.1.0",
    description="U-Net land-cover segmentation on Sentinel-2 imagery with PGD attack simulation and FGSM adversarial defense.",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "pillow>=10.0",
    ],
)
