from setuptools import setup
requirement = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="evfs",
    version="0.1",
    description="evolutionary feature search",
    url="git@github.com:cloudwalk/simple-evolutionary-feature-search.git",
    license="MIT",
    install_requires=requirement,
    packages=["evfs"],
    zip_safe=False,
)
