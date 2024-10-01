// Function to initialize the WebSocket and handle sending/receiving frames
function initializeWebSocket(onOpenCallback) {
    const ws = new WebSocket('ws://20.172.32.213:8080/ws'); // Adjust the WebSocket server URL if needed

    ws.onopen = () => {
        console.log('WebSocket connection opened.');
        if (typeof onOpenCallback === 'function') {
            onOpenCallback(ws); // Execute callback when WebSocket is ready
        }
    };

    ws.onclose = () => {
        console.log('WebSocket connection closed.');
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    return ws;
}

// Function to capture video stream from the user's camera
async function captureCameraStream() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const localVideo = document.getElementById('localVideo');
        localVideo.srcObject = stream;

        const videoTrack = stream.getVideoTracks()[0];
        return videoTrack;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

// Function to send video frames over WebSocket
function sendFramesOverWebSocket(ws, track) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    const videoElement = document.createElement('video');
    videoElement.srcObject = new MediaStream([track]);

    videoElement.onloadedmetadata = () => {
        videoElement.play();
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        console.log("Video dimensions set:", canvas.width, canvas.height);

        const captureFrame = () => {
            if (ws.readyState === WebSocket.OPEN) { // Ensure WebSocket is open
                context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                console.log('Captured frame from camera and drew on canvas');

                canvas.toBlob((blob) => {
                    if (blob) {
                        console.log("Sending frame to server...");
                        blob.arrayBuffer().then((buffer) => {
                            console.log(`Frame size: ${buffer.byteLength} bytes`);
                            ws.send(buffer); // Send the frame as binary data
                        });
                    } else {
                        console.log("Failed to capture frame as blob");
                    }
                }, 'image/jpeg');

                // Capture and send frame every 100ms
                setTimeout(captureFrame, 100);
            } else {
                console.log('WebSocket not open yet, retrying in 100ms');
                setTimeout(captureFrame, 100); // Retry after 100ms
            }
        };

        captureFrame();
    };
}

// Function to receive processed frames from WebSocket and display them
function receiveFramesFromWebSocket(ws) {
    const remoteVideo = document.getElementById('remoteVideo');

    ws.onmessage = (event) => {
        console.log("Received processed frame from server");

        // Create a blob from the binary data received
        const blob = new Blob([event.data], { type: 'image/jpeg' });
        const imageUrl = URL.createObjectURL(blob);

        // Create an image element to display the received frame
        const img = new Image();
        img.src = imageUrl;

        img.onload = () => {
            // Clear the previous frame and display the new one
            remoteVideo.src = '';
            remoteVideo.srcObject = null;
            remoteVideo.src = imageUrl;
            console.log("Updated remote video with new frame");
        };

        img.onerror = (error) => {
            console.error("Failed to load image:", error);
        };
    };
}

// Main function to initialize the camera stream and WebSocket
async function startCameraStream() {
    const videoTrack = await captureCameraStream();
    if (videoTrack) {
        const ws = initializeWebSocket((ws) => {
            sendFramesOverWebSocket(ws, videoTrack);
            receiveFramesFromWebSocket(ws);
        });
    }
}

// Start the camera stream and WebSocket connection when the page loads
window.onload = () => {
    startCameraStream();
};
