#!/bin/zsh

# Default input device
INPUT_DEVICE=${1:-"BlackHole"}

HOST=193.106.93.127
PORT=43007
SAMPLING_RATE=16000
OUTPUT_FILE="audio_output_$(date +"%Y%m%d_%H%M%S").wav"

# Check if input device is supported
if [[ "$INPUT_DEVICE" == "BlackHole" ]]; then
    DEVICE_ID="BlackHole"
elif [[ "$INPUT_DEVICE" == "Microphone" ]]; then
    DEVICE_ID="MacBook Air Microphone"
else
    echo "Unknown input device: $INPUT_DEVICE"
    echo "Usage: $0 [BlackHole|Microphone]"
    exit 1
fi

# Function to handle SIGPIPE and prevent SoX from exiting
trap '' SIGPIPE

# Function to handle SIGINT (Ctrl+C) and properly stop recording
cleanup() {
    echo "Stopping the recording and finalizing the file..."
    pkill -P $$
    wait
    echo "Recording stopped and file finalized."
    exit 0
}

# Trap SIGINT (Ctrl+C) to call the cleanup function
trap cleanup SIGINT

# Start recording to a file in the background
#sox -t coreaudio "$DEVICE_ID" -r $SAMPLING_RATE -c 1 "$OUTPUT_FILE" &

# Start streaming over the network
while true; do
    echo "Waiting for receiver to open connection..."
    
    if nc -z "$HOST" "$PORT"; then
        echo "Connection established. Starting streaming from $INPUT_DEVICE..."
        
        sox -t coreaudio "$DEVICE_ID" -r $SAMPLING_RATE -c 1 -b 16 -t raw - | \
        stdbuf -oL socat -u - TCP:"$HOST":"$PORT",retry=30,interval=5
        
        # Check the exit status of the pipeline
        if [[ $? -eq 0 ]]; then
            echo "Streaming finished successfully. Exiting..."
            exit 0
        else
            echo "Connection was broken or an error occurred. Retrying in 5 seconds..."
            sleep 5
        fi
    else
        echo "Receiver not ready. Retrying in 5 seconds..."
        sleep 5
    fi
done

