from setuptools import setup, find_packages

setup(
    name="numax_proxies",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "lightkurve",
    ],
    python_requires=">=3.12",
    description="A Python tool to compute νmax proxiess.",
    url="https://github.com/Johanneshj/numax_proxies",
    author="Johannes Jørgensen",
    author_email="johannes.joergensen@uibk.ac.at",
    license="MIT",
)
