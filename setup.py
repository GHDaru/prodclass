from setuptools import setup, find_packages

# Correção para leitura do README.md com a codificação UTF-8
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prodclass",
    version="0.1.6",  # Arquivos de exemplos
    packages=find_packages(include=['prodclass',"prodclass.*"]),
    # package_data={
    #     'prodclass': ['examples/*.csv'],  # Inclui os arquivos CSV na pasta data dentro do pacote prodclass
    # },
    include_package_data=True,  # Isso diz ao setuptools para incluir os arquivos definidos em MANIFEST.in
    install_requires=[
        "pandas>=1.2",
        "numpy>=1.19",
        "scikit-learn>=0.24",  # Descomente se você estiver usando scikit-learn no seu projeto
        "matplotlib>=3.3",  # Opcional, dependendo do uso
        "seaborn>=0.11",  # Opcional, dependendo do uso
        "statsmodels>=0.12"  # Opcional, dependendo do uso
    ],
    # Meta-dados
    author="Gilsiley Henrique Darú",
    author_email="ghdaru@gmail.com",
    description="Uma biblioteca Python para auxiliar na vetorização e categorização de descrições de produto.  Possui benchmarks argmax e machine learning embutidos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GHDaru/prodclass/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
