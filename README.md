This is how to run the voice agent server in in the dev environment.

### Setup the environment

1. Copy your `id_rsa.pub` into `docker` folder to ssh directly into containers.
2. Make sure ports and volumes are valid in `docker/docker-compose-dev.yml`.
3. Run `up-dev.sh`. It will build a docker image and run a container with shh daemon and jupyter server.
4. If you don't see `voice-agent-core-container-dev` in `docker ps` output, check `docker-compose logs` for errors.
5. Try to ssh into container from e.g. your laptop:  
`ssh user@<hostname> -p <port>`  
where `<port>` is the port specified in `docker-compose-dev.yml`.
6. Install dependencies:  
```bash
cd sources/voice-agent-core
./install.sh
```

### Setup the server

1. The server uses secure web socket connection and relies on Caddy, which nicely manages SSL certificates for us. On your host machine, make sure you have proper config in the Caddyfile (usually `/etc/caddy/Caddyfile`):  
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