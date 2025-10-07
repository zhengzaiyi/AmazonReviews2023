#!/usr/bin/bash

# Source file: https://github.sc-corp.net/Snapchat/research/blob/main/deployment/gce_experiment_instance/startup_v4.sh
# Uploaded to GCS: gs://research-monorepo-dev/gce_experiment_instance/startup_v4.sh

# This script setus up a GCE instance w/:
# 1. `snapaccess`  - so your VM's attached SA can authenticate against ATS to access GHE, internal registry, etc.
# 2. `git snap` - so you can access repos hosted on Snap's internal GHE
# 3. Misc dependencies to make experimentation / dev easier.
#
#
# Installation Instructions:
# It can be insalled by running the following command:
# bash -c "$(gsutil cat gs://research-monorepo-dev/gce_experiment_instance/startup_v4.sh)"
#
# This is a modified fork of:https://github.sc-corp.net/Snapchat/research/blob/a993cba7fedfc2905e6dfb6d2996d45b139b8118/deployment/gce_experiment_instance/startup_v3.sh
# Specifically, instead of running this on startup which is quite complicated to maintain and test - we ask users to run this manually
# once they have created a new instance.

set -e
set -x

username=$(whoami)

sa_account=$(gcloud config list account --format "value(core.account)")

INTERNAL_SNAP_BIN="/home/$username/internal/snap"

ASSET_BUCKET="gs://research-monorepo-dev"
SC_CHIMERA_WHL_PATH="$ASSET_BUCKET/gce_experiment_instance/pkgs/sc_chimera-1.0.500413-py2.py3-none-any.whl"
LCA_WHL_PATH="$ASSET_BUCKET/gce_experiment_instance/pkgs/lca-1.10.415.tar.gz"
DEPENDENCY_REGISTRY_SETUP_SCRIPT_PATH="gs://dependency-registry-setup/general/latest/setup.sh"
MAMBA_PY_ENV_NAME="python38"
MAMBA_PY_ENV_VERSION="3.8"
MAMBA_RELEASE="25.3.1-0" # https://github.com/conda-forge/miniforge/releases/tag/25.3.1-0
REQUIRED_MAKE_VERSION="4.4"
INSTALL_GNU_MAKE_VERSION="4.4.1"


has_mamba_installed() {
    if ! mamba --version > /dev/null 2>&1; then
        echo "Mamba not found, will try again, trying to see if it is installed but not initialized"
        if ! source "/opt/conda/etc/profile.d/conda.sh" || ! source "/opt/conda/etc/profile.d/mamba.sh"; then
            echo "Failed to find mamba"
            return 1
        fi
        if ! mamba --version > /dev/null 2>&1; then
            echo "WARNING: Failed to initialize mamba correctly. Script may fail."
            return 1
        fi
    fi
    echo "Mamba found in the current shell"
    return 0
}

# Check if we have permission to read from $ASSET_BUCKET
if ! gsutil cp $SC_CHIMERA_WHL_PATH /home/$username/ >/dev/null 2>&1; then
    echo "ERROR: You do not have permission to read from $ASSET_BUCKET."
    echo "Please ensure service account: $sa_account has 'roles/storage.objectViewer', and 'roles/storage.legacyBucketReader' access to $ASSET_BUCKET."
    exit 1
fi

if ! gsutil cp $DEPENDENCY_REGISTRY_SETUP_SCRIPT_PATH /home/$username/ >/dev/null 2>&1; then
    echo "ERROR: You do not have permission to read from $DEPENDENCY_REGISTRY_BUCKET."
    echo "Please ask #appsec-guests to allowlist $sa_account to the 'dependency-registry~reader' ACL group: https://lease.sc-corp.net/v2/view_iam?resourceType=ACL_GROUP&parentResource=dependency-registry&resource=reader"
    exit 1
fi

# [SETUP GCOMMON DEV TOOLS]
sudo apt-get update

# Install essential development tools and utilities:
# build-essential: GNU C/C++ compiler (gcc/g++)
# make: Powerful automation tool that reads 'Makefile' files to manage and compile software projects.
# unzip: Simple but necessary utility for decompressing files from the common .zip archive format.
# podman: Alternate to docker, daemonless container engine that allows you to build, run, and manage OCI-compatible containers.
sudo apt-get install -y build-essential make unzip podman


echo "Required make version: $REQUIRED_MAKE_VERSION"
echo "Current make version: $CURRENT_MAKE_VERSION"

# Function to compare semantic versions; using sort -V to compare semantic versions seems to be the most reliable
get_min_make_version() {
    if [ "$1" != "$2" ]; then
        printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1
    else
        echo "$1"
    fi
}

is_running_on_mac() {
    [ "$(uname)" == "Darwin" ]
    return $?
}

echo "Checking make version..."
# Make > v4.4 has features like `.WAIT` which is useful for projects; thus we install it here specifically.
# Currently, some Ubuntu images dont have access to make > v4.3 even through apt-get install make.
# Thus we need to install it from source.
current_make_version=$(make --version | head -n 1 | awk '{print $3}')
if [ "$(get_min_make_version "$current_make_version" "$REQUIRED_MAKE_VERSION")" != "$REQUIRED_MAKE_VERSION" ]; then
    echo "Current make version ($current_make_version) is less than ($REQUIRED_MAKE_VERSION)"
    echo "Installing GNU MAKE v${INSTALL_GNU_MAKE_VERSION}"
    if is_running_on_mac; then
        brew install make
        echo "The brew formula shoulda have installed GNU “make” as “gmake”. See: https://formulae.brew.sh/formula/make"
        echo "You should be able to use 'gmake' inplace of wherever you use 'make' now."
    else
        wget https://ftp.gnu.org/gnu/make/make-${INSTALL_GNU_MAKE_VERSION}.tar.gz
        tar xvf  make-${INSTALL_GNU_MAKE_VERSION}.tar.gz
        cd make-${INSTALL_GNU_MAKE_VERSION}
        ./configure # Setup required deps
        make # Generate the relevant resources
        if ! command -v sudo 2>&1 >/dev/null
        then
            echo "Sudo not found, this is expected if we're being built by docker."
            make install # Install the new make version
        else
            echo "Sudo found, installing make with sudo."
            sudo make install # Install the new make version
        fi
        cd ..
        rm -rf make-${INSTALL_GNU_MAKE_VERSION}
        rm make-${INSTALL_GNU_MAKE_VERSION}.tar.gz
    fi
else
    echo "Current make version ($current_make_version) is greater than or equal to ($REQUIRED_MAKE_VERSION)"
fi


# We still install docker for backwards compatibility of older projects
if ! command -v docker >/dev/null 2>&1; then
    if ! grep -qi 'ubuntu' /etc/os-release; then
        echo "ERROR: Docker not installed, and running on non-ubuntu OS. Please install docker manually."
        echo "Will exit now."
        exit 1
    fi

    # Docker install instructions: https://docs.docker.com/engine/install/ubuntu/
    # Uninstall all conflicting packages, if they exist
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
        sudo apt-get remove $pkg;
    done

    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    # Install the Docker packages.
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# Permission to be able to run docker commands
sudo usermod -aG docker $username
# If we have a docker socket already running, we need to change the permissions
sudo chmod 777 /var/run/docker.sock

# Ensure we can run multi-arch docker builds
docker buildx create --driver=docker-container --use
sudo apt-get install -y qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Opinionated decision to install zsh as well - its better than bash
if ! command -v zsh >/dev/null 2>&1; then
    echo "Zsh is not installed. Installing zsh..."
    sudo apt install -y zsh
    echo "Installing oh-my-zsh: https://ohmyz.sh/#install"
    # RUNZSH=no → prevents the script from launching zsh right after install
    # CHSH=yes → changes default shell to zsh
    RUNZSH=no CHSH=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    # This is for convenience, as if you ever install tools like coursier, they will write to your non-interactive shells
    # files like .profile, so you will have to manually source your `.profile` file in your interactive shells.
    [ -f ~/.zprofile ] || touch ~/.zprofile
    grep -q 'source .*\.profile' ~/.zshrc || \
    echo '[[ -e ~/.profile ]] && source ~/.profile' | cat - ~/.zshrc > /tmp/zshrc && mv /tmp/zshrc ~/.zshrc
fi
# ========== [SETUP COMMON DEV TOOLS]


# [SETUP PYTHON ENV FOR USE BY SNAP SCRIPTS]
if ! has_mamba_installed; then
    echo "Mamba is not installed. Installing miniforge..."
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_RELEASE}/Miniforge3-$(uname)-$(uname -m).sh"
    sudo bash Miniforge3.sh -b -p "/opt/conda" # This is the default place where conda-forge enabled GCP images install conda/mamba, etc

    # Source mamba and conda so we can use it in the current shell
    source "/opt/conda/etc/profile.d/conda.sh"
    source "/opt/conda/etc/profile.d/mamba.sh" # Mamba support

    # Initialize conda and mamba for zsh, and bash
    # We initialize conda for backwards compatibility
    conda init bash
    echo 'eval "$(mamba shell hook --shell bash)"' >> ~/.bashrc
    conda init zsh
    echo 'eval "$(mamba shell hook --shell zsh)"' >> ~/.zshrc
fi

# Allow all users to read/write to conda dir to manage environments
sudo chmod -R 755 /opt/conda/
sudo chmod 777 /opt/conda/pkgs/
sudo chmod 777 /opt/conda/envs/

# Activate mamba environment so we can use pip, python, etc
mamba activate

# Install python3.8 and prepare env
# crontab needed from set up kernel auto update
pip install python-crontab

# create python 3.8
# TODO: (svij) Fix this, mamba permissions error:
# critical libmamba filesystem error: cannot set permissions: Operation not permitted [/opt/conda/pkgs/cache]
# mamba create -y -n $MAMBA_PY_ENV_NAME python=$MAMBA_PY_ENV_VERSION
# Check where mamba stores its envs
# mamba_env_dir=$(mamba info | awk -F': ' '/envs directories/{print $2}')
conda create -y --override-channels --channel conda-forge -n $MAMBA_PY_ENV_NAME python=$MAMBA_PY_ENV_VERSION
mamba_env_dir=$(conda info | awk -F': ' '/envs directories/{print $2}')

# install necessary libs
$mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/pip install wheel
$mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/pip install ipykernel
$mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/pip install oauth2client

# ========== [SETUP PYTHON ENV FOR USE BY SNAP SCRIPTS]


# [SETUP LCA] - used for authenticating against Snap Systems
# install lca first as it is needed for creating token for ATS
gsutil cp $SC_CHIMERA_WHL_PATH /home/$username/
gsutil cp $LCA_WHL_PATH /home/$username/
$mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/pip install /home/$username/sc_chimera-1.0.500413-py2.py3-none-any.whl
$mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/pip install /home/$username/lca-1.10.415.tar.gz
rm /home/$username/sc_chimera-1.0.500413-py2.py3-none-any.whl
rm /home/$username/lca-1.10.415.tar.gz

# Create centralized $INTERNAL_SNAP_BIN/ directory to house common files
sudo mkdir -p $INTERNAL_SNAP_BIN/
sudo chmod 777 $INTERNAL_SNAP_BIN/

# Install registry credentials helper
gsutil cp gs://dependency-registry-setup/general/latest/setup.sh $INTERNAL_SNAP_BIN/.registry_setup.sh


cat <<EOF > $INTERNAL_SNAP_BIN/.snapaccess_wrapper
#!/bin/bash
# This is just a simple wrapper around registry_setup script below to enable
# using the same command locally and on cloud to get credentials to access internal registry.
# The actual snapaccess tool is not available currently for cloud instances.
# See https://wiki.sc-corp.net/display/SEC/Dependency+Registry+User+Guide

if [[ "\$1" == "credentials" && "\$2" == "refresh" ]] ; then
    bash $INTERNAL_SNAP_BIN/.registry_setup.sh
fi
EOF
sudo chmod +x $INTERNAL_SNAP_BIN/.snapaccess_wrapper
sudo ln -s $INTERNAL_SNAP_BIN/.snapaccess_wrapper /usr/local/bin/snapaccess


# register the kernel in jupyterlab
$mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/python -m ipykernel install --user --name $MAMBA_PY_ENV_NAME --display-name "Python ($MAMBA_PY_ENV_NAME)"

# ========== [SETUP LCA]


# [SETUP GHE ACCESS]

# Firstly add script that can add ATS related headers to git commands
# https://docs.google.com/document/d/1g9-hhJj_9y08y4AQZ1iwwinb8Lx34Ae1trXG3qXZ1Ss/edit#heading=h.4rzt5nlyflw4
cat <<EOF > $INTERNAL_SNAP_BIN/.git_snap
# Automatically added from startup script to make it easier to access GHE

py_script_gen_lca_token="
import lca
from oauth2client.client import GoogleCredentials
import subprocess

credentials = GoogleCredentials.get_application_default()
sa_account = (
    subprocess.check_output("'"gcloud config list account --format \"value(core.account)\""'", shell=True)
    .decode(\"utf-8\")
    .strip()
)
issuer = lca.LcaIssuer(sa_account, credentials)
token = issuer.get_lca_token(\"ats.snap\", 300)
print(token)
"

LCA_TOKEN=\$($mamba_env_dir/$MAMBA_PY_ENV_NAME/bin/python3 -c "\$py_script_gen_lca_token")
git -c http.extraHeader="SC-LCA-1:\$LCA_TOKEN" \\
    -c http.extraHeader="x-ats-integration-id: ghe-cli" \\
    -c http.extraHeader="Host: ats-ingress-us-central1-gcp.api.snapchat.com" \\
    \$@;
EOF

# Update gitconfig, so its easier to call git commands through ATS
cat <<EOF > $INTERNAL_SNAP_BIN/.gitconfig
# Automatically added from startup script to make it easier to access GHE
[url "https://ingress-us-central1-gcp.api.snapchat.com/"]
    insteadOf = git@github.sc-corp.net:

[url "https://ingress-us-central1-gcp.api.snapchat.com/"]
    insteadOf = https://github.sc-corp.net/

[alias]
    snap = "!sh $INTERNAL_SNAP_BIN/.git_snap"

# End of automatically added script
EOF

cat $INTERNAL_SNAP_BIN/.gitconfig >> ${HOME}/.gitconfig
chown $username:$username ${HOME}/.gitconfig


git config --global user.email "$username@snapchat.com"
git config --global user.name "$username"

# ========== [SETUP GHE ACCESS]

# Update permissions on $INTERNAL_SNAP_BIN
sudo chmod -R 777 $INTERNAL_SNAP_BIN

echo "Done! Please restart your machine to ensure setup is complete"