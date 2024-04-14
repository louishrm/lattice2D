from setuptools import setup, find_packages

setup(
    name='Lattice2D',  # the name of your package
    version='0.1',  # the current version of your package
    packages=find_packages(),  # automatically discover all packages and subpackages. Alternatively, you can specify the package names manually as a list.
    description='Python package for working with 2D crystals',  # a brief description of your package
    url='https://github.com/louishrm/lattice2D',  # the URL for your package's homepage, typically the GitHub repository
    author='Louis Sharma',  # your name
    author_email='louis.sharma2303@gmail.com',  # your email address
    #license='LICENSE.txt',  # the license for the package
    install_requires=['numpy', 'matplotlib'],  # a list of other Python packages that your package depends on
)