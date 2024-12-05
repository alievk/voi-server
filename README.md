### Setup the environment

1. Copy your `id_rsa.pub` into `docker` folder to ssh directly into the container.
2. Make sure ports and volumes are valid in `docker/docker-compose.yml`. Check `${HOME}/.gitconfig`.
3. Run `up.sh`. It will build a docker image and run a container with a shh  and jupyter server.
4. If you don't see `voice-agent-core-container` in `docker ps`, check `docker-compose logs` for errors.
5. Try to ssh into container from e.g. your laptop:  
`ssh user@<hostname> -p <port>`  
with the `<port>` specified in `docker-compose.yml`. When ssh-ed you will see a prompt like this:  
`user@044cce00d859:~/sources/voice-agent-core$`
6. Run `./install.sh` from the above working path.

### Setup the server

1. The server uses secure web socket connection and relies on Caddy, which nicely manages SSL certificates for us. Make sure you have proper config in the Caddyfile (usually `/etc/caddy/Caddyfile`):  
```
your_domain.com:8764 {
    reverse_proxy localhost:8765
}
```
This will proxy secure web socket connection from `8765` to `8764` port.
1. Set `LITELLM_API_BASE` and `LITELLM_API_KEY` variables in `.env` file.
2. Check paths in:
- `tts_models.json` (checkpoints)
- `agents.json` (cached audios) 
3. Run server `python3 simple_server.py`.