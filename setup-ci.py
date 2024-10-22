from setuptools import setup, find_namespace_packages
import codecs


def parse_requirements(requirements_path='./requirements.txt'):
    """Recursively parse requirements from nested pip files."""
    install_requires = []
    with codecs.open(requirements_path, 'r') as handle:
        # remove comments and empty lines
        lines = (line.strip() for line in handle
                 if line.strip() and not line.startswith('#'))

        for line in lines:
            # check for nested requirements files
            if line.startswith('-r'):
                # recursively call this function
                install_requires += parse_requirements(req_path=line[3:])

            else:
                # add the line as a new requirement
                install_requires.append(line)

    return install_requires


with open('build_version.txt') as f:
    build = str(f.readline())


setup(
    name='databricks-llm-fine-tuning-demo',
    version=build,
    install_requires=parse_requirements('./requirements-ci.txt'),
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    url='',
    license='',
    author='Northell',
    description='Northell content extraction solution',
    include_package_data=True
)


with open('build_version.txt', "r+") as f:
    data = f.read()
    f.seek(0)
    parts = build.split(".")
    revision = parts[len(parts)-1]
    f.write(str(int(revision) +1))
    f.truncate()
