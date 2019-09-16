#!/usr/bin/sh
set -e

check_python_version() {
    printf "Checking for python3.7... "
    if ! [ -x "$(command -v python3.7)" ]; then
        printf "\nERROR: Python 3.7 not found. You must install Python 3.7 before running this script.\n" >&2
        exit 1
    else
        printf "OK!\n"
    fi
}

check_python_header() {
    printf "Checking for Python.h (python3.7-dev)... "
    # first, makes sure distutils.sysconfig usable
    if ! $(python -c "import distutils.sysconfig" &> /dev/null); then
        printf "\nERROR: distutils.sysconfig not usable\n" >&2
        exit 3
    fi
    # get include path for this python version
    INCLUDEPY=$(python3.7 -c "from distutils import sysconfig as s; print(s.get_config_var('INCLUDEPY'))")
    if [ ! -f "${INCLUDEPY}/Python.h" ]; then
        printf "Not found!\n"
        printf "\nERROR: The python development headers are not installed. Aborting...\n" >&2
        exit 4
    else
        printf "OK!\n"
    fi
}

check_git_lfs() {
    printf "Checking for git lfs... "
    if ! [ -x "$(command -v git-lfs)" ]; then
        printf "Will be installed.\n"
        BOOTSTRAP_GIT_LFS=true
    else
        printf "OK!\n"
        BOOTSTRAP_GIT_LFS=false
    fi
}


check_cmake(){
    printf "Checking for cmake... "
    if ! [ -x "$(command -v cmake)" ]; then
        printf "Will be installed.\n"
        BOOTSTRAP_CMAKE=true
    else
        printf "OK!\n"
        BOOTSTRAP_CMAKE=false
    fi
}

check_gdal_config() {
    printf "Checking for gdal-config (gdal-dev)... "
    if ! [ -x "$(command -v gdal-config)" ]; then
        if ! [ "$1" = "--include-gdal" ]; then
            printf "Not found!\n"
            local MSG="\nERROR: gdal-config was not found. "
            MSG+="You may install gdal-dev using your system's package manager "
            MSG+="or you may pass the --include-gdal to the install script.\n"
            echo -e $MSG >&2
            exit 5
        else
            printf "Will be installed.\n"
            INCLUDE_GDAL="--include-gdal=True"
        fi
    else
        printf "OK!\n"
        if [ "$1" = "--include-gdal" ]; then
            if [ "$2" = "--no-confirm" ] || [ "$2" = "--noconfirm" ]; then
                INCLUDE_GDAL="--include-gdal=True"
            else
                local CONFIM_GDAL=false
                echo
                local MSG=""
                MSG+="gdal-config was found on the system but you passed the --include-gdal option."
                MSG+="Are you sure you want to override the system's GDAL and install a local version?"
                MSG+="You may use the --no-confirm flag to avoid this confirmation prompt."
                MSG+="([y]es or [N]o): "
                read -p "$MSG"
                case $(echo $REPLY | tr '[A-Z]' '[a-z]') in
                    y|yes) INCLUDE_GDAL="--include-gdal=True" ;;
                    *) INCLUDE_GDAL="" ;;
                esac
            fi
        fi
    fi
}

make_virtual_env() {
   python3.7 -m venv .geomesh_env 
}

source_virtual_env() {
    source .geomesh_env/bin/activate
}

upgrade_pip() {
    pip install --upgrade pip
}

bootstrap_git_lfs() {
    if [ BOOTSTRAP_GIT_LFS=true ]; then
        curl -L https://github.com/git-lfs/git-lfs/releases/download/v2.8.0/git-lfs-linux-amd64-v2.8.0.tar.gz | tar xz -C .geomesh_env/bin "git-lfs"
    fi
    git lfs install
}

bootstrap_cmake() {
    if [ BOOTSTRAP_CMAKE=true ]; then
        pip install cmake
    fi
}

bootstrap_deps() {
    ./setup.py install_deps $INCLUDE_GDAL
}

bootstrap_install() {
    ./setup.py install
}

make_alias() {
    local PYENV_ALIAS="alias geomesh=\"source $(pwd)/.geomesh_env/bin/activate\""
    if [[ $SHELL = *zsh ]]; then
        if ! grep -Fxq "$PYENV_ALIAS" ~/.zshrc; then
            echo "# ------ Added by geomesh installer" >> ~/.zshrc
            echo "$PYENV_ALIAS" >> ~/.zshrc
        fi
    elif [[ $SHELL = *bash ]]; then
        if ! grep -Fxq "$PYENV_ALIAS" ~/.bashrc; then
            echo "# ------ Added by geomesh installer" >> ~/.bashrc
            echo "$PYENV_ALIAS" >> ~/.bashrc
        fi
    fi

}


exit_msg() {
    local MSG="\n"
    MSG+="geomesh has been installed. "
    MSG+="You must source the python virtual enviroment by executing the alias:\n"
    MSG+="\ngeomesh\n"
    MSG+="\nYou must restart your shell in order for this command to be available.\n\n"
    MSG+="Alternatively, you may manually source the geomesh python virtual enviroment by executing:\n"
    MSG+="\nsource $(pwd)/.geomesh_env/bin/activate\n\n"
    MSG+="Happy coding!\n"
    echo -e $MSG
}

main() {
    check_python_version
    check_python_header
    check_git_lfs
    check_cmake
    check_gdal_config "$@"
    make_virtual_env
    source_virtual_env
    upgrade_pip
    bootstrap_git_lfs
    bootstrap_cmake
    bootstrap_deps
    bootstrap_install
    make_alias
    exit_msg
}

main "$@"
