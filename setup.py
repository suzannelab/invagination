import os
import sys
import subprocess

from setuptools import setup
from setuptools import find_packages


DISTNAME = "invagination"
DESCRIPTION = "mesoderm invagination modeling based on tyssue"
MAINTAINER = "Guillaume Gay"
MAINTAINER_EMAIL = "guillaume@damcb.com"
URL = "https://github.com/suzannelab/invagination"
LICENSE = "GPL-v3.0"
DOWNLOAD_URL = "https://github.com/suzannelab/invagination.git"

files = ["notebooks/*.*", "data/*.*"]

## Version management copied form numpy
## Thanks to them!
MAJOR = 0
MINOR = 0
MICRO = 2
ISRELEASED = True
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip().decode("ascii")
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of tyssue.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists(".git"):
        git_revision = git_version()
    elif os.path.exists("invagination/version.py"):
        # must be a source distribution, use existing version file
        # read from it instead of importing to avoid importing
        # the whole package
        with open("invagination/version.py", "r", encoding="utf-8") as version:
            for line in version.readlines():
                if line.startswith("git_revision"):
                    git_revision = line.split("=")[-1][2:-2]
                    break
    else:
        git_revision = "Unknown"

    if not ISRELEASED:
        FULLVERSION += ".dev0+" + git_revision[:7]

    return FULLVERSION, git_revision


def write_version_py(filename="invagination/version.py"):
    cnt = """
# THIS FILE IS GENERATED FROM invagination SETUP.PY
#
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, "w")
    try:
        a.write(
            cnt
            % {
                "version": VERSION,
                "full_version": FULLVERSION,
                "git_revision": GIT_REVISION,
                "isrelease": str(ISRELEASED),
            }
        )
    finally:
        a.close()


if __name__ == "__main__":

    write_version_py()
    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,
        #classifiers=[
        #    "Development Status :: 4 - Beta",
        #    "Intended Audience :: Science/Research",
        #    "License :: OSI Approved :: MPL v2.0",
        #    "Natural Language :: English",
        #    "Operating System :: MacOS",
        #    "Operating System :: Microsoft",
        #    "Operating System :: POSIX :: Linux",
        #    "Programming Language :: Python :: 3.6",
        #    "Topic :: Scientific/Engineering :: Bio-Informatics",
        #    "Topic :: Scientific/Engineering :: Medical Science Apps",
        #],
        packages=find_packages(),
        package_data={"invagination": files},
        include_package_data=True,
        zip_safe=False,
    )
