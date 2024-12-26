import asyncio
import json
from datetime import datetime
import signal
import os
from threading import Lock

import torch
import websockets
from loguru import logger

from recognition import OnlineASR
from generation import MultiVoiceGenerator, DummyVoiceGenerator, AsyncVoiceGenerator
from audio import AudioOutputStream, AudioInputStream, WavSaver, convert_f32le_to_s16le
from llm import get_agent_config, stringify_content, ConversationContext, BaseLLMAgent, CharacterLLMAgent, CharacterEchoAgent
from conversation import Conversation

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


async def send_error(websocket, error_message):
    try:
        await websocket.send(serialize_message({"type": "error", "error": error_message}))
    except Exception as e:
        logger.error(f"Error sending error message: {e}")


async def handle_connection(websocket):
    try:
        await handle_conversation(websocket)
    except Exception as e:
        await send_error(websocket, f"Error in handle_connection: {e}")


async def handle_conversation(websocket):
    log_dir = f"logs/conversations/{get_timestamp()}"
    logger.add(f"{log_dir}/server.log", rotation="100 MB")

    logger.info("New connection is started")

    async def voice_generator_error_handler(error):
        await send_error(websocket, f"Error in VoiceGenerator: {error}")

    async def conversation_error_handler(error):
        await send_error(websocket, f"Error in Conversation: {error}")

    @logger.catch(reraise=True)
    async def handle_input_audio(audio_chunk):
        # audio_chunk is f32le
        await conversation.handle_input_audio(audio_chunk)
        if audio_chunk is not None:
            audio_input_saver.write(audio_chunk)

    @logger.catch(reraise=True)
    async def handle_context_changed(context):
        messages = context.get_messages(filter=lambda msg: not msg["handled"])
        for msg in messages:
            data = {
                "type": "message",
                "role": msg["role"],
                "content": stringify_content(msg["content"]),
                "time": msg["time"].strftime("%H:%M:%S"),
                "id": msg["id"]
            }
            logger.info("Sending message: {}", data)
            try:
                await websocket.send(serialize_message(data))
                context.update_message(msg["id"], handled=True)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    @logger.catch(reraise=True)
    async def handle_generated_audio(audio_chunk, speech_id, duration=None):
        # audio_chunk is f32le
        if audio_chunk is None: # end of audio
            conversation.on_assistant_audio_end(speech_id, duration)
        else:
            metadata = {
                "type": "audio",
                "speech_id": str(speech_id)
            }
            try:
                s16le_chunk = convert_f32le_to_s16le(audio_chunk)
                await websocket.send(serialize_message(metadata, s16le_chunk))
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")

            audio_output_saver.write(audio_chunk) # TODO: once per 1Mb it flushes buffer (blocking)

    async def get_validated_init_message():
        try:
            data = await websocket.recv()
            init_message = json.loads(data)
            logger.info("Received init message: {}", init_message)

            if "agent_name" not in init_message:
                raise ValueError("Missing required field 'agent_name'")

            agent_config = get_agent_config(init_message["agent_name"])
            if not agent_config:
                raise KeyError(f"Agent '{init_message['agent_name']}' not found")

            return agent_config

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            error_msg = f"Init message error: {str(e)}"
            logger.error(error_msg)
            await send_error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            await send_error(error_msg)
            return None

    logger.info("Waiting for init message")
    agent_config = await get_validated_init_message()
    if not agent_config:
        return

    logger.info("Initializing speech recognition")
    asr = OnlineASR(
        cached=True
    )

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
        output_chunk_size_ms=1000,
        input_sample_rate=16000, # client sends 16000
        output_sample_rate=asr.sample_rate,
        input_format="pcm16"
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
        error_cb=conversation_error_handler
    )

    logger.info("Initializing audio input saver")
    audio_input_saver = WavSaver(
        f"{log_dir}/incoming.wav", 
        sample_rate=audio_input_stream.output_sample_rate
    )

    logger.info("Initializing audio output saver")
    audio_output_saver = WavSaver(
        f"{log_dir}/outgoing.wav", 
        sample_rate=voice_generator.sample_rate
    )

    await websocket.send(serialize_message({"type": "init_done"}))

    conversation.greeting()

    logger.info("Entering main loop")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                logger.debug(f"Received audio chunk: {len(message)} bytes")
                if not audio_input_stream.is_running():
                    audio_input_stream.start()
                audio_input_stream.put(message)
            else:
                try:
                    message_data = json.loads(message)
                    message_type = message_data["type"]
                    if message_type == "start_recording":
                        logger.info("Received start_recording message")
                        audio_input_stream.start()
                    elif message_type in ["create_response", "stop_recording"]:
                        logger.info("Received stop_recording message")
                        audio_input_stream.stop()
                    elif message_type == "manual_text":
                        logger.info("Received manual_text message")
                        conversation.on_manual_text(message_data["content"])
                    elif message_type == "interrupt":
                        logger.info("Received interrupt message")
                        conversation.on_user_interrupt(
                            speech_id=int(message_data["speech_id"]), 
                            interrupted_at=message_data["interrupted_at"])
                    else:
                        e = f"Received unexpected type: {message_type}"
                        logger.warning(e)
                        await send_error(e)
                except Exception as e:
                    e = f"Error processing message {message}: {e}"
                    logger.warning(e)
                    await send_error(e)
        logger.info("WebSocket connection closed, checking timeout")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed unexpectedly")
    finally:
        logger.info("Cleaning up resources")
        voice_generator.stop()
        audio_input_stream.stop()
        audio_input_saver.close()
        audio_output_saver.close()

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


async def main():
    logger.add(f"logs/server/server-{get_timestamp()}.log", rotation="100 MB")

    loop = asyncio.get_running_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )

    server = await websockets.serve(
        handle_connection,
        "0.0.0.0",
        8765,
    )
    logger.info("WebSocket server started on wss://0.0.0.0:8765")

    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        logger.info("Server is shutting down...")
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
