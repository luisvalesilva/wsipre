import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wsipre",
    version="0.1.0",
    author="Luis A. Vale Silva",
    author_email="luisvalesilva@gmail.com",
    description="Whole-slide image preprocessing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    keywords="whole-slide image virtual slide OpenSlide machine learning deep learning",
    url="https://github.com/luisvalesilva/wsipre",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
