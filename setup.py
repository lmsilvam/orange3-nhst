from setuptools import setup, find_namespace_packages

setup(
    name="Orange3-NHST",
    version="0.1.0",
    description="Orange add-on: classical NHST tests (t-tests, one-way ANOVA).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    keywords=("orange3 add-on", "t-test", "anova", "nhst"),
    python_requires=">=3.11",
    # implicit namespace package: orangecontrib has no __init__.py
    packages=find_namespace_packages(include=["orangecontrib", "orangecontrib.*"]),
    package_data={"orangecontrib.nhst.widgets": ["icons/*.svg"]},
    install_requires=["Orange3>=3.41.0", "numpy", "scipy"],
    extras_require={"test": ["pytest"]},
    entry_points={
        "orange3.addon": ["nhst = orangecontrib.nhst"],
        "orange.widgets": ["Statistics (NHST) = orangecontrib.nhst.widgets"],
        "orange.canvas.help": [
            "html-index = orangecontrib.nhst.widgets:WIDGET_HELP_PATH"
        ],
    },
    zip_safe=False,
)
