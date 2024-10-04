// Function to initialize the WebSocket and handle sending/receiving frames
function initializeWebSocket(onOpenCallback) {
    const ws = new WebSocket('ws://20.172.32.213:8080/ws'); // Adjust the WebSocket server URL if needed

    ws.onopen = () => {
        console.log('WebSocket connection opened.');
        if (typeof onOpenCallback === 'function') {
            onOpenCallback(ws); // Execute callback when WebSocket is ready
        }
    };

    ws.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    return ws;
}

// Function to populate the video source selector
async function populateVideoSources() {
    try {
        // Request permission to access media devices
        await navigator.mediaDevices.getUserMedia({ video: true, audio: false });

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        const videoSourceSelect = document.getElementById('videoSource');
        videoSourceSelect.innerHTML = ''; // Clear any existing entries

        videoDevices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${videoSourceSelect.length + 1}`;
            videoSourceSelect.appendChild(option);
        });

        // Listen for changes to the selected camera
        videoSourceSelect.onchange = () => {
            // Stop any existing stream
            if (window.currentStream) {
                window.currentStream.getTracks().forEach(track => track.stop());
            }
            // Restart the camera stream with the new selection
            startCameraStream();
        };
    } catch (error) {
        console.error('Error enumerating devices:', error);
    }
}

// Function to capture video stream from the user's selected camera
async function captureCameraStream() {
    try {
        const videoSourceSelect = document.getElementById('videoSource');
        const selectedDeviceId = videoSourceSelect.value;

        const constraints = {
            video: {
                width: { ideal: 512 },
                height: { ideal: 512 },
                deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
            },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        const localVideo = document.getElementById('localVideo');
        localVideo.width = 512;
        localVideo.height = 512;
        localVideo.srcObject = stream;

        // Store the stream globally to stop it later if needed
        window.currentStream = stream;

        return stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

// Function to send video frames over WebSocket
function sendFramesOverWebSocket(ws, stream) {
    const videoElement = document.createElement('video');
    videoElement.srcObject = stream;
    videoElement.width = 512;
    videoElement.height = 512;
    videoElement.muted = true;
    videoElement.play();

    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const context = canvas.getContext('2d');

    let lastFrameTime = 0;
    const targetFPS = 3;
    const frameInterval = 1000 / targetFPS;


    const captureFrame = (timestamp) => {
        if (ws.readyState === WebSocket.OPEN) { // Ensure WebSocket is open
            const elapsed = timestamp - lastFrameTime;
            if (elapsed > frameInterval) {
                lastFrameTime = timestamp;

                context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    if (blob) {
                        blob.arrayBuffer().then((buffer) => {
                            ws.send(buffer); // Send the frame as binary data
                        });
                    } else {
                        console.log("Failed to capture frame as blob");
                    }
                }, 'image/jpeg', 0.7); // Adjust the quality parameter as needed
            }
        } else {
            console.log('WebSocket not open, cannot send frame.');
        }

        requestAnimationFrame(captureFrame);
    };

    videoElement.oncanplay = () => {
        requestAnimationFrame(captureFrame);
    };
}

// Function to receive processed frames from WebSocket and display them
function receiveFramesFromWebSocket(ws) {
    const remoteCanvas = document.getElementById('remoteVideo');
    const context = remoteCanvas.getContext('2d');

    ws.onmessage = (event) => {
        console.log("received frame");
        const blob = new Blob([event.data], { type: 'image/jpeg' });
        const img = new Image();

        img.onload = () => {
            context.drawImage(img, 0, 0, remoteCanvas.width, remoteCanvas.height);
            URL.revokeObjectURL(img.src); // Clean up the object URL
        };

        img.onerror = (err) => {
            console.error('Image load error:', err);
            URL.revokeObjectURL(img.src);
        };

        img.src = URL.createObjectURL(blob);
    };
}


// Main function to initialize the camera stream and WebSocket
async function startCameraStream() {
    const stream = await captureCameraStream();
    if (stream) {
        const ws = initializeWebSocket((ws) => {
            sendFramesOverWebSocket(ws, stream);
            receiveFramesFromWebSocket(ws);
        });
    }
}

// Start the camera stream and WebSocket connection when the page loads
window.onload = async () => {
    await populateVideoSources();
    await startCameraStream();
};
