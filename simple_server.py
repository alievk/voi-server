import asyncio
import websockets
import ssl
import logging
import ffmpeg

import sys, os
sys.path.append(os.path.dirname(__file__))
import voicebot_audio_api.src.main as vba

vba.CACHE_ROOT = '/home/user/.cache'
model = vba.get_whisper_model()

logging.basicConfig(level=logging.DEBUG)

# Global variable to store audio chunks
audio_chunks = []

async def handle_audio(websocket, path):
    global audio_chunks
    logging.info(f"New connection from {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                logging.info(f"Received audio chunk: {len(message)} bytes")
                audio_chunks.append(message)
            elif message == 'STOP_RECORDING':
                logging.info("Received stop recording signal")
                if audio_chunks:
                    await transcribe(websocket, audio_chunks)
                    audio_chunks = []  # Clear the chunks after saving
            else:
                logging.info(f"Received text message: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        logging.info(f"Connection closed: {e}")
    except Exception as e:
        logging.error(f"Error in audio handler: {e}", exc_info=True)
    finally:
        audio_chunks = []  # Clear the chunks after saving

async def transcribe(websocket, chunks):
    filename = save_audio_to_file(chunks)
    iterator = vba.async_transcribe(path=filename, model=model, language='ru')

    skip_first = False
    try:
        async for text_chunk in iterator:
            if not skip_first:
                skip_first = True
                continue
            text = text_chunk['text']
            logging.info(text)
            await websocket.send(text)
    except Exception as e:
        error_message = "Unexpected error while processing audio"
        await websocket.send(error_message)
        raise e

def save_audio_to_file(chunks):
    # Combine all chunks into a single byte string
    audio_data = b''.join(chunks)
    
    try:
        # Use ffmpeg to convert the WebM audio to WAV
        process = (
            ffmpeg
            .input('pipe:0', format='webm')  # Specify input format as WebM
            .output('recorded_audio.wav', format='wav', acodec='pcm_s16le', ar=48000, ac=1)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        
        stdout, stderr = process.communicate(input=audio_data)
        
        if process.returncode == 0:
            logging.info("Audio saved to recorded_audio.wav")
        else:
            logging.error(f"FFmpeg error: {stderr.decode()}")
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {str(e)}")
    except Exception as e:
        logging.error(f"Error saving audio: {str(e)}")

    return "recorded_audio.wav"

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('localhost+2.pem', 'localhost+2-key.pem')

async def main():
    server = await websockets.serve(
        handle_audio,
        "0.0.0.0",
        8765,
        ssl=ssl_context
    )
    logging.info("WebSocket server started on wss://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())