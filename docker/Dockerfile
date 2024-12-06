FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install SSH server, sudo, and other dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    netcat \
    sudo \
    curl \
    git \
    vim \
    htop \
    tmux \
    ffmpeg \
    python3-pip \
    && mkdir /var/run/sshd

# Create a non-root user with the same UID and GID as the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} user && \
    useradd -m -u ${USER_ID} -g user -s /bin/bash user

# Add the user to the sudo group
RUN usermod -aG sudo user

# Allow passwordless sudo for the user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up SSH configuration
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Copy public key for SSH
RUN mkdir /home/user/.ssh
COPY id_rsa.pub /home/user/.ssh/authorized_keys
RUN chmod 700 /home/user/.ssh && chmod 600 /home/user/.ssh/authorized_keys

# Install essential Python packages
RUN pip3 install \
    jupyterlab==4.2.5 \
    ipywidgets

# Set up Jupyter configuration for the new user
RUN mkdir -p /home/user/.jupyter
COPY jupyter_lab_config.py /home/user/.jupyter/

# Create Trash directory and set permissions
RUN mkdir -p /home/user/.local/share/Trash

# Install NVM
ENV NVM_DIR="/usr/local/nvm"
ENV NODE_VERSION="18"
RUN mkdir -p "$NVM_DIR" \
    && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install $NODE_VERSION \
    && NODE_ACTUAL_VERSION=$(ls -d $NVM_DIR/versions/node/v$NODE_VERSION.* | tail -n 1 | xargs basename) \
    && ln -s $NVM_DIR/versions/node/$NODE_ACTUAL_VERSION/bin/node /usr/local/bin/node \
    && ln -s $NVM_DIR/versions/node/$NODE_ACTUAL_VERSION/bin/npm /usr/local/bin/npm \
    && ln -s $NVM_DIR/versions/node/$NODE_ACTUAL_VERSION/bin/npx /usr/local/bin/npx

# Ensure npm global config and PATH are set for all users
RUN echo 'export NVM_DIR="/usr/local/nvm"' >> /etc/bash.bashrc \
    && echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> /etc/bash.bashrc \
    && echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> /etc/bash.bashrc \
    && echo 'export NPM_CONFIG_PREFIX="$HOME/.npm-global"' >> /etc/bash.bashrc \
    && echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> /etc/bash.bashrc \
    && echo 'mkdir -p "$HOME/.npm-global"' >> /etc/bash.bashrc

# Add local bin to PATH
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/bash.bashrc

# Copy start script and make it executable for the new user
COPY start.sh start-dev.sh /home/user/
RUN chmod +x /home/user/start.sh /home/user/start-dev.sh

# Fix permissions
RUN chown -R user:user /home/user

USER user
WORKDIR /home/user