from setuptools import setup


with open('README.md') as fp:
    readme = fp.read()

setup(
    name='heavytailed',
    version='1.0.1',
    description='Perform distribution analysis on heavy-tailed distributed data',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Xiangwen Wang',
    author_email='wangxiangwen1989@gmail.com',
    url='https://github.com/XiangwenWang/heavytailed',
    keywords='heavy-tailed, distribution analysis',
    packages=['heavytailed'],
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'pandas', 'mpmath', 'matplotlib'],
    license="BSD 2-Clause License",
    zip_safe=True,
    platforms='any',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)
