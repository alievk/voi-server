<p align="center">
  <img src="https://github.com/user-attachments/assets/f4f571a1-b06f-4c38-8cb6-88d6ca54d19f" width="256" height="256" />
</p>

Voi is a free and open source backend for realtime voice agents. Check the JS [client](https://github.com/alievk/voi-js-client).

# Table of Contents
- [Requirements](#requirements)
  - [Hardware](#hardware)
  - [Software](#software)
- [Setup](#setup)
  - [Development environment](#development-environment)
  - [Caddy](#caddy)
  - [LiteLLM](#litellm)
  - [Voi server](#voi-server)
    - [Environment variables](#environment-variables)
    - [Speech-to-text model](#speech-to-text-model)
    - [Text-to-speech models](#text-to-speech-models)
    - [Custom text-to-speech models](#custom-text-to-speech-models)
    - [Agents](#agents)
    - [Run the server](#run-the-server)
  - [Access tokens](#access-tokens)
  - [Philosophy](#philosophy)
  - [Contribution](#contribution)
  - [License](#license)

## Requirements

### Hardware
- 9Gb+ of GPU memory. I recommend GeForce RTX 3090 or better for a single worker.
- 8-core CPU, 32Gb RAM is enough.
- 10Gb of disk space.

### Software
- Ubuntu 22.04 or higher.
- Fresh Nvidia drivers (tested on 545+ driver versions).
- Docker with Nvidia runtime support.
- Caddy server.
- LiteLLM server.

## Setup
Voi uses Docker Compose to run a server. It uses Docker mostly for runtime, while keeping source code, Python packages and model weights on the host file system. This was made intentionally to allow fast development. 

There are two Docker environments to run Voi server, production and development. Basically they are the same, except the production config starts the Voi server automatically and uses a different port.

### Development environment

Get the sources.
```bash
git clone https://github.com/alievk/voi-core.git
cd voi-core
```

Copy your `id_rsa.pub` into `docker` folder to be able to ssh directly into the container.
```bash
cp ~/.ssh/id_rsa.pub docker/
```

Make a copy of `docker/docker-compose-dev.example.yml`.
```bash
cp docker/docker-compose-dev.example.yml docker/docker-compose-dev.yml
```
In the `docker-compose-dev.yml`, edit the `environment`, `ports` and `volumes` sections as you need. If you need a Jupyter server, set your token in the `JUPYTER_TOKEN` variable, otherwise it won't run for safety reasons.

Build and run the development container.
```bash
cd docker
./up-dev.sh
```

When the container is created, you will see `voice-agent-core-container-dev` in `docker ps` output, otherwise check `docker-compose logs` for errors. If there were no errors, ssh daemon and Jupyter server will be listening on the ports defined in `docker-compose-dev.yml`.

Connect via ssh into the container from e.g. your laptop:  
```bash
ssh user@<host> -p <port>
```  
where `<host>` is the address of your host machine and `port` is the port specified in `docker-compose-dev.yml`. You will see a bash prompt like `user@8846788f5e9c:~$`.

My personal recommendation is to add a config to your `~/.ssh/config` file to easily connect to the container:
```
Host voi_docker_dev
  Hostname your_host_address
  AddKeysToAgent yes
  UseKeychain yes
  User user
  Port port_from_the_above
```

Then you do just this and get into the container:
```bash
ssh voi_docker_dev
```

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

### LiteLLM
LiteLLM allows calling all LLM APIs using OpenAI format, which is neat.  

#### Restricted regions
If you run a Voi server in a country restricted by OpenAI (like Russia or China), you will need to run a remote LiteLLM server in a closest unrestricted country. You can do this for just $10/mo using AWS Lightsail. These are minimal specs you need:
- 2 GB RAM, 2 vCPUs, 60 GB SSD
- Ubuntu

If you use AWS Lightsail, do not forget to add a custom TCP rule for the port 4000.

If you are not in the restricted region, you can run LiteLLM server locally on your host machine.

#### Setup
For the details of setting up LiteLLM, visit [the repo](https://github.com/BerriAI/litellm), but basically you need to follow these steps.

Get the code.
```bash
git clone https://github.com/BerriAI/litellm
cd litellm
```  

Add the master key - you can change this after setup.
```bash
echo 'LITELLM_MASTER_KEY="sk-1234"' > .env
source .env
```  

Create models configuration file.
```bash
vim litellm_config.yaml
```  

Example configuration:
```yaml
model_list:
  - model_name: gemini-1.5-flash
    litellm_params:
      model: openai/gemini-1.5-flash
      api_key: your_googleapi_key
      api_base: https://generativelanguage.googleapis.com/v1beta/openai
  - model_name: meta-llama-3.1-70b-instruct-turbo
    litellm_params:
      model: openai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
      api_key: your_deepinfra_key
      api_base: https://api.deepinfra.com/v1/openai
```

The `model` format is `{API format}/{model name}`, where `{API format}` is `openai`/`anthropic` and `{model name}` is the model name in the provider's format (`gpt-4o-mini` for OpenAI or `meta-llama/Meta-Llama-3.1-8B-Instruct` for DeepInfra). Look at [litellm_config.example.yaml](litellm_config.example.yaml) for more examples.  

- Start LiteLLM server.
```bash
docker-compose up
```

### Voi server
Before running the server, we need to set the environment variables and create agents config.  

My typical workflow is to run the [development environment]() and ssh to the container using Cursor (`Connect to Host` -> `voi_docker_dev`). In this way, I can edit source code and run the scripts in one place.

#### Environment variables
Make a copy of `.env.example`.
```bash
# Assuming you are in the Voi root
cp .env.example .env
```

- `LITELLM_API_BASE` is the address of your LiteLLM server, like `http://111.1.1.1:4000` of `http://localhost:4000`.
- `LITELLM_API_KEY` is `LITELLM_MASTER_KEY` from the LiteLLM's `.env` file.
- `TOKEN_SECRET_KEY` is a secret key for generating [access tokens](#access-tokens) to the websocket endpoint. You should not reveal this key to a client.
- `API_KEY` is the HTTPS API access key. You need to share it with a client.

#### Speech-to-text model
Voi relies on [Whisper](https://github.com/openai/whisper) for speech transcribition and adds realtime (transcribe-as-you-speak) processing on top of that. The model weights are downloaded automatically on the first launch.

#### Text-to-speech models
Voi uses xTTS-v2 model to generate speech. It gives the best tradeoff between the quality and speed.  

To test your agents, you can download the pre-trained multi-speaker model from [HuggingFace](https://huggingface.co/coqui/XTTS-v2/tree/main). Download these files and put them in a directory of your choice (f.e., `models/xtts_v2`):
- `model.pth`
- `config.json`
- `vocab.json`
- `speakers_xtts.pth`

Then make a copy of `tts_models.example.json` and fix the paths in `multispeaker_original` so that they point to the model files above.
```bash
cp tts_models.example.json tts_models.json
```

#### Custom text-to-speech models
Voi allows changing voice tone of the agent dynamically during the conversation (like neutral or excited), but the pre-trained model coming along with xTTS doesn't allow this. I have a custom pipeline for fine-tuning text-to-speech models on audio datasets and enabling dynamic tone changing, which I'm not open sourcing today. If you need a custom model, please DM me on [X](https://x.com/alievk0).

#### Agents
`agents.json` is where you create your virtual personalities, define their behaviour and voice. `agents.example.json` demonstrates some fun use cases and utility agents like a shallow speech-to-text test and a sub-agent for discarding Llama's "I can't create explicit content"-like responses.

Make a copy of `agents.example.json` and edit it for your needs.
```bash
cp agents.example.json agents.json
```

Each agent in `agents.json` has the following structure:
- `llm_model`: The language model to use (must match models in `litellm_config.yaml`)
- `control_agent` (optional): Name of an agent that filters/controls the main agent's responses
- `voices`: Configuration for speech synthesis
  - `character`: Main voice settings
    - `model`: TTS model name from `tts_models.json`
    - `voice`: Voice identifier for the model
    - `speed` (optional): Speech speed multiplier
  - `narrator` (optional): Voice for narrative comments
    - Same settings as `character` plus:
    - `leading_silence`: Silence before narration
    - `trailing_silence`: Silence after narration
- `system_prompt`: Array of strings defining the agent's personality and behavior. Can include special templates:
  - `{character_agent_message_format_voice_tone}`: Adds instructions for voice tone control (neutral/warm/excited/sad)
  - `{character_agent_message_format_narrator_comments}`: Adds instructions for narrator comments format (*actions in third person*)
- `examples` (optional): List of conversation examples for few-shot learning
- `greetings`: Initial messages configuration
  - `choices`: List of greeting messages (can include pre-cached voice files)
  - `voice_tone`: Emotional tone for greeting (must match tones in `tts_models.json`)

Special agents like `control_agent` can have additional fields:
- `model`: Processing type (e.g. "pattern_matching")
- `denial_phrases`: Phrases to filter out
- `giveup_after`: Number of retries before giving up
- `giveup_response`: Fallback responses

#### Run the server
Ssh into the container and run:
```bash
python3 ws_server.py
```

Note that the first time the client connects to an agent it may take some time to load the text-to-speech models.

### Access tokens
Clients and agents communicate via the websocket. A client must receive it's personal token to access the websocket endpoint. This can be made in two ways:
1. Through the API:
```bash
curl -I -X POST "https://your_host_address:port/integrations/your_app" \
-H "API-Key: your_api_key"
```
where 
- `your_host_address:port` is the address of the host running Voi server and `port` is the port where the server is listening.
- `your_app` is the name of your app, like `relationships_coach`.
- `your_api_key` is `API_KEY` from `.env`.

Note that this will generate a token which will expire after 1 day.

2. Manually:
```bash
python3 token_generator.py your_app --expire n_days
```
Here you can set any number of `n_days` when your token will expire.

## Philosophy
Realtime conversation with a human is a really complex task, as it requires from the agent an empathy, competence and speed. If you lack a single piece of these, your agent is useless. That's why making a good voice agent is not just stacking a bunch of APIs together. You have to develop it very carefully, making a small step, then testing, making a small step, then testing...  

There are two main factors which enabled me to run this project. First, the emergence of smart, fast and cheap LLMs necessary for agents intelligence. Second, the advancement of code copilots. Though I have a deep learning background, there are lots of topics beyond my competence required to build a good voice agent.

While open sourcing Voi, I realized many people could use it for learning software engineering. Yes, this is still actual, because this project is basically many pieces of AI-generated code carefully stitched together by a human engineer.

## Contribution
You are welcome to open PRs with bug fixes, new features and documentation improvements.

Or you can just buy me a coffee and I will convert it to code!  
[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](buymeacoffee.com/alievk)

## License
Voi uses the MIT license, which basically means you can do anything with it, free of charge. However, the dependencies may have different licenses. Check them if you care.
