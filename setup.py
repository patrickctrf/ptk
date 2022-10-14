import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Se nao houver requirements file, crie com os requisitos
requirements = 'numpy\ntorch\ntensorflow\n'

try:
    # Trazemos os requirements atualizados do arquivo txt
    with open("requirements.txt", "r") as file:
        requirements_txt = file.read().splitlines()
except IOError:
    # Se nao houver arquivo, criamos com o minimo necessario
    with open("requirements.txt", "w") as file:
        file.write(requirements)
except FileNotFoundError:
    # Se nao houver arquivo, criamos com o minimo necessario
    with open("requirements.txt", "w") as file:
        file.write(requirements)
finally:
    # Realizamos a leitura do arquivo de uma forma ou outra, para seguir com o setup
    with open("requirements.txt", "r") as file:
        requirements_txt = file.read().splitlines()


setuptools.setup(
    name="ptk-patrickctrf",
    version="0.1.2",
    author="patrickctrf",
    author_email="patrickctrf@gmail.com",
    description="Package for Python programming by Patrick Ferreira",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patrickctrf/ptk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements_txt
)

# Para gerar uma string do REQUIREMENTS.TXT:
#with open("requirements.txt", "r") as file:
    # A funcao abaixo exibe na tela do terminal a string a colocar no inicio DESTE arquivo
#    file.read()
