import asyncio
import json
from datetime import datetime
import signal
import os
from threading import Lock
import jwt
from fastapi import FastAPI, WebSocket, HTTPException, Security, Depends
import uvicorn
from loguru import logger
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

import torch

from recognition import OnlineASR
from generation import MultiVoiceGenerator, DummyVoiceGenerator, AsyncVoiceGenerator
from audio import AudioInputStream, WavGroupSaver, convert_f32le_to_s16le, convert_s16le_to_ogg
from llm import agent_config_manager, ConversationContext, BaseLLMAgent, CharacterLLMAgent, CharacterEchoAgent, voice_tone_emoji
from conversation import Conversation
from token_generator import generate_token, TOKEN_SECRET_KEY

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

cuda_lock = Lock()


def serialize_message(metadata, blob=None):
    metadata = json.dumps(metadata).encode('utf-8')
    metadata_length = len(metadata).to_bytes(4, byteorder='big')
    if blob is None:
        return metadata_length + metadata
    else:
        return metadata_length + metadata + blob


async def safe_send(websocket, message, fatal=False):
    try:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_text(message)
    except RuntimeError as e:
        if fatal:
            raise e
        logger.error(f"Error sending message: {e}")


async def send_error(websocket, error_message):
    await safe_send(websocket, serialize_message({
        "type": "error", 
        "error": error_message
    }))


def validate_token(websocket):
    query_string = websocket.scope['query_string'].decode('utf-8').split('?')[-1]
    params = dict(param.split('=') for param in query_string.split('&'))
    token = params.get('token')
    if not token:
        raise ValueError("Token is required")

    try:
        decoded_jwt = jwt.decode(token, TOKEN_SECRET_KEY, algorithms=['HS256'])
        if decoded_jwt['expire'] < datetime.now().isoformat():
            raise ValueError("Token expired")
    except jwt.InvalidTokenError as e:
        raise ValueError("Invalid token")

    return decoded_jwt


async def handle_connection(websocket):
    try:
        token_data = validate_token(websocket)
    except Exception as e:
        e = f"Token validation error: {e}"
        logger.error(e)
        await send_error(websocket, e)
        return

    try:
        await start_conversation(websocket, token_data)
    except Exception as e:
        await send_error(websocket, f"Error in start_conversation: {repr(e)}")
        raise e


async def start_conversation(websocket, token_data):
    app_id = token_data["app"]
    user_speech_counter = 0
    assistant_speech_counter = 0

    log_dir = f"logs/conversations/{app_id}/{get_timestamp()}"
    logger.add(f"{log_dir}/server.log", rotation="100 MB")

    logger.info("New connection is started")

    async def voice_generator_error_handler(error):
        await send_error(websocket, f"Error in VoiceGenerator: {error}")

    async def conversation_error_handler(error):
        await send_error(websocket, f"Error in Conversation: {error}")

    @logger.catch(reraise=True)
    async def handle_input_audio(audio_chunk):
        # audio_chunk is f32le
        nonlocal user_speech_counter
        await conversation.handle_input_audio(audio_chunk)
        if audio_chunk is None:
            user_audio_saver.close(f"user_{user_speech_counter}.wav")
            user_speech_counter += 1
        else:
            user_audio_saver.write(audio_chunk, f"user_{user_speech_counter}.wav")

    @logger.catch(reraise=True)
    async def handle_context_changed(context):
        messages = context.get_messages(filter=lambda msg: not msg["handled"])
        for msg in messages:
            data = {
                "type": "message",
                "role": msg["role"],
                "time": msg["time"].strftime("%H:%M:%S"),
                "id": msg["id"],
                "from": msg.get("from", "text")
            }
            if isinstance(msg["content"], dict):
                data["content"] = msg["content"]["text"]
                data["sentiment"] = voice_tone_emoji(msg["content"]["voice_tone"])
            else:
                data["content"] = msg["content"]

            logger.info("Sending message: {}", data)
            await safe_send(websocket, serialize_message(data))
            context.update_message(msg["id"], handled=True)

    @logger.catch(reraise=True)
    async def handle_generated_audio(audio_chunk, speech_id, duration=None):
        # audio_chunk is f32le
        nonlocal assistant_speech_counter
        stream_output_audio = init_message.get("stream_output_audio", True)
        file_id = f"assistant_{assistant_speech_counter}.wav"

        payload = None
        if audio_chunk is None: # end of audio
            full_audio = assistant_audio_saver.get_buffer(file_id)
            if full_audio and not stream_output_audio: # send full audio
                payload = convert_s16le_to_ogg(
                    full_audio,
                    sample_rate=voice_generator.sample_rate
                )

            conversation.on_assistant_audio_end(speech_id, duration)
            assistant_audio_saver.close(file_id)
            assistant_speech_counter += 1
        else:
            s16le_chunk = convert_f32le_to_s16le(audio_chunk)
            if stream_output_audio:
                payload = s16le_chunk

            assistant_audio_saver.write(audio_chunk, file_id)

        if payload:
            metadata = {
                "type": "audio",
                "speech_id": str(speech_id)
            }
            await safe_send(websocket, serialize_message(metadata, payload))

    async def get_validated_init_message():
        known_fields = [
            "type", 
            "agent_name", 
            "agent_config",
            "stream_user_stt",
            "final_stt_correction",
            "stream_output_audio", 
            "input_audio_format",
            "init_greeting"
        ]
        required_fields = [
            "type", 
            "agent_name"
        ]

        init_message = await websocket.receive_json()
        logger.info("Received init message: {}", init_message)

        for field in init_message:
            if field not in known_fields:
                raise ValueError(f"Unknown field '{field}'")

        for field in required_fields:
            if field not in init_message:
                raise ValueError(f"Missing required field '{field}'")

        if init_message["type"] != "init":
            raise ValueError("Invalid message type")

        return init_message

    async def invoke_llm(message_data):
        try:
            if not "model" in message_data:
                raise ValueError("Missing required field 'model'")
            if not "prompt" in message_data:
                raise ValueError("Missing required field 'prompt'")
            if not "messages" in message_data:
                raise ValueError("Missing required field 'messages'")

            logger.debug(f"Invoking LLM with message: {message_data}")
            agent = BaseLLMAgent(
                model_name=message_data["model"],
                system_prompt=message_data["prompt"]
            )
            context = ConversationContext(messages=message_data["messages"])
            llm_response = agent.completion(context)
            message = {
                "type": "llm_response",
                "content": llm_response
            }
            await safe_send(websocket, serialize_message(message))
        except Exception as e:
            error = f"Error invoking LLM: {e}"
            logger.error(error)
            await send_error(websocket, error)

    logger.info("Waiting for init message")
    init_message = await get_validated_init_message()
    if not init_message:
        return

    logger.info("Initializing speech recognition")
    asr = OnlineASR(
        cached=True
    )

    if init_message.get("agent_config"):
        agent_config_manager.add_agent(init_message["agent_name"], json.loads(init_message["agent_config"]))
    agent_config = agent_config_manager.get_config(init_message["agent_name"])

    logger.info("Initializing voice generation")
    if "voices" not in agent_config:
        voice_generator = DummyVoiceGenerator()
    else:
        voice_generator = MultiVoiceGenerator.from_config(
            agent_config["voices"],
            cached=True
        )
    voice_generator = AsyncVoiceGenerator(
        voice_generator, 
        generated_audio_cb=handle_generated_audio,
        error_cb=voice_generator_error_handler
    )
    voice_generator.start()

    logger.info("Initializing audio input stream")
    audio_input_stream = AudioInputStream(
        handle_input_audio,
        output_chunk_size_ms=500,
        input_sample_rate=16000, # client sends 16000
        output_sample_rate=asr.sample_rate,
        input_format=init_message.get("input_audio_format", "pcm16")
    )
    audio_input_stream.start()

    logger.info("Initializing response agent")
    if agent_config["llm_model"] == "echo":
        character_agent = CharacterEchoAgent()
    else:
        character_agent = CharacterLLMAgent.from_config(agent_config)

    conversation_context = ConversationContext(
        context_changed_cb=handle_context_changed
    )
    
    logger.info("Initializing conversation")
    conversation = Conversation(
        asr=asr,
        voice_generator=voice_generator,
        character_agent=character_agent,
        conversation_context=conversation_context,
        stream_user_stt=init_message.get("stream_user_stt", True),
        final_stt_correction=init_message.get("final_stt_correction", True),
        error_cb=conversation_error_handler
    )

    logger.info("Initializing audio saver")
    user_audio_saver = WavGroupSaver(
        f"{log_dir}/audio", 
        sample_rate=audio_input_stream.output_sample_rate
    )
    assistant_audio_saver = WavGroupSaver(
        f"{log_dir}/audio", 
        sample_rate=voice_generator.sample_rate
    )

    await safe_send(websocket, serialize_message({
        "type": "init_done",
        "agent_name": init_message["agent_name"]
    }), fatal=True)

    if init_message.get("init_greeting", True):
        conversation.greeting()

    logger.info("Entering main loop")
    while True:
        try:
            message = await websocket.receive()
        except RuntimeError as e:
            logger.error(f"Error receiving message: {e}")
            break

        if message["type"] == "websocket.disconnect":
            logger.info(f"WebSocket disconnected with code {message.get('code')}")
            break
        
        if "bytes" in message:
            if not audio_input_stream.is_running():
                audio_input_stream.start()
            audio_input_stream.put(message["bytes"])
        else:
            try:
                message_data = json.loads(message["text"])
                message_type = message_data["type"]
                logger.debug(f"Received message: {message_data}")
                if message_type == "start_recording":
                    audio_input_stream.start()
                elif message_type in ["create_response", "stop_recording"]:
                    audio_input_stream.stop()
                elif message_type == "manual_text":
                    conversation.on_manual_text(message_data["content"])
                elif message_type == "interrupt":
                    conversation.on_user_interrupt(
                        speech_id=int(message_data["speech_id"]), 
                        interrupted_at=message_data["interrupted_at"])
                elif message_type == "invoke_llm":
                    asyncio.run_coroutine_threadsafe(invoke_llm(message_data), asyncio.get_running_loop())
                else:
                    e = f"Message type {message_type} is not supported"
                    logger.warning(e)
                    await send_error(websocket, e)
            except Exception as e:
                e = f"Error processing client message: {e}. Message: {message}"
                logger.error(e)
                await send_error(websocket, e)

    logger.info("Cleaning up resources")
    voice_generator.stop()
    audio_input_stream.stop()
    user_audio_saver.close()
    assistant_audio_saver.close()

    with cuda_lock:
        torch.cuda.empty_cache()
        
    logger.info("Connection is done")


def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


async def shutdown(signal, loop):
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="API-Key")
active_connections = 0

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    return {
        "active_connections": active_connections
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections
    active_connections += 1
    await websocket.accept()
    try:
        await handle_connection(websocket)
    finally:
        active_connections = max(0, active_connections - 1)

@app.post("/integrations/{app_id}")
async def create_integration_token(
    app_id: str,
    api_key: str = Depends(verify_api_key)
):
    try:
        token = generate_token(TOKEN_SECRET_KEY, app_id)
        return {"token": token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def main():
    logger.add(f"logs/server/server-{get_timestamp()}.log", rotation="100 MB")

    loop = asyncio.get_running_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )

    config = uvicorn.Config(app, host="0.0.0.0", port=8765)
    server = uvicorn.Server(config)
    
    logger.info("Server started on http://0.0.0.0:8765")
    
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Server is shutting down...")
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
