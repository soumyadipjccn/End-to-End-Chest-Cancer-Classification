import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "End-to-End-Chest-Cancer-Classification"
AUTHOR_USER_NAME = "ml-engineer"
SRC_REPO = "chestCancerClassifier"
AUTHOR_EMAIL = "admin@chestcancerclassifier.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="End-to-End Chest Cancer Classification using Deep Learning, MLflow, DVC, Flask and Docker",
    long_description=long_description,
    long_description_content_policy="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
