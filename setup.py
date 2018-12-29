import setuptools

exec(open('fastai_tf_fit/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastai-tf-fit",
    version = __version__,
    author="Bryan Heffernan",
    license = "Apache License 2.0",
    description="Fit Tensorflow/Keras models with fastai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = 'fastai, deep learning, machine learning, pytorch, keras, tensorflow',
    url="https://github.com/fastai/tf-fit",
    packages=setuptools.find_packages(),
    install_requires=['fastai==1.0.39', 'tensorflow'],
    python_requires  = '==3.6.*',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
)
