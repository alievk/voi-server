# Voi
Voi is a free and open source backend for reatime voice agents. Check the JS [client](https://github.com/alievk/voi-js-client).

## Requrements

### Hardware
- 9Gb+ of GPU memory. I recommend GeForce RTX 3090 or better for a single worker.
- 8-core CPU, 32Gb RAM is enough.
- 10Gb of disk space.

### Software
- Ubuntu 22.04 or higher.
- Fresh Nvidia drivers (tested on 545+ driver versions).
- Docker with Nvidia runtime support.
- Caddy server.

## Setup
Voi uses Docker Compose to run a server. It uses Docker mostly for runtime, while keeping source code, Python packages and model weights on the host file system. This was made intentionally to allow fast development. 

There are two Docker environments to run Voi server, production and development. Basically they are the same, except the production config starts the Voi server automatically and uses a different port.

### Development environment

Get the sources.
```
git clone https://github.com/alievk/voi-core.git
cd voi-core
```

Copy your `id_rsa.pub` into `docker` folder to be able to ssh directily into the container.
```
cp ~/.ssh/id_rsa.pub docker/
```

Make a copy of `docker/jupyter_lab_config.example.py`. Follow the instructions inside this file if you need a password for Jupyter or leave it as is.
```
cp docker/jupyter_lab_config.example.py docker/jupyter_lab_config.py
```

Make a copy of `docker/docker-compose-dev.example.yml`.
```
cp docker/docker-compose-dev.example.yml docker/docker-compose-dev.yml
```
In the `docker-compose-dev.yml`, edit the `environment`, `ports` and `volumes` sections as you need.

Build and run the development container.
```
cd docker
./up-dev.sh
```

When the container is created, you will see `voice-agent-core-container-dev` in `docker ps` output, otherwise check `docker-compose logs` for errors. If there were no errors, ssh daemon and Jupyter server will be listening on the ports defined in `docker-compose-dev.yml`.

Connect via ssh into the container from e.g. your laptop:  
```
ssh user@<host> -p <port>
```  
where `<host>` is the address of your host machine and `port` is the port specified in `docker-compose-dev.yml`. You will see a bash prompt like `user@8846788f5e9c:~$`.

In the container, install the Python dependencies:  
```bash
cd voi-core
./install.sh
```
This step is intentionally not incorporated in the Dockerfile because at the active development stage you often change the requirements and don't want to rebuild the container each time. You won't need to do this each time when the container was restarted if you have mapped `.local` directory properly in the `docker-compose-dev.yml`.

### Caddy
The Voi server uses secure web socket connection and relies on Caddy, which nicely manages SSL certificates for us. Follow [the docs](https://caddyserver.com/docs/install) to get it.  

On your host machine, make sure you have proper config in the Caddyfile (usually `/etc/caddy/Caddyfile`):  
```
your_domain.com:8774 {
    reverse_proxy localhost:8775
}
```
This will proxy secure web socket connection from `8775` to `8774` port.  


1. Set `LITELLM_API_BASE` and `LITELLM_API_KEY` variables in `.env` file.
2. Check paths in:
- `tts_models.json` (checkpoints)
- `agents.json` (cached audios) 
3. Run server `python3 simple_server.py`.
