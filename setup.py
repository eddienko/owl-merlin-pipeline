from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = [
    "dask",
    "distributed",
    "owl-pipeline-develop",
]

setup_requirements = ["pytest-runner", "flake8"]

test_requirements = ["coverage", "pytest", "pytest-cov", "pytest-mock"]


setup(
    author="Eduardo Gonzalez Solares",
    author_email="e.gonzalezsolares@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Owl Shell Pipeline",
    entry_points={"owl.pipelines": "merlin = merlin_pipeline"},
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords="imaxt, owl",
    name="owl-merlin-pipeline",
    packages=find_packages(include=["merlin_pipeline*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://eddienko.github.com/owl-merlin-pipeline",
    version="0.1.0",
    zip_safe=False,
    python_requires=">=3.7",
)
