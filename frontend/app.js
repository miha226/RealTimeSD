// Attach currentTargetFPS to the window object for global access
window.currentTargetFPS = 6; // Default FPS

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

    const captureFrame = (timestamp) => {
        const currentFrameInterval = 1000 / window.currentTargetFPS;
        // console.log(`Current FPS: ${window.currentTargetFPS}, Frame Interval: ${currentFrameInterval.toFixed(2)} ms`);

        if (ws.readyState === WebSocket.OPEN) { // Ensure WebSocket is open
            const elapsed = timestamp - lastFrameTime;
            if (elapsed > currentFrameInterval) {
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
            // console.log('WebSocket not open, cannot send frame.');
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
        // console.log("Received frame");
        const blob = new Blob([event.data], { type: 'image/jpeg' });
        const img = new Image();
        console.log(`Received frame at ${Date.now() / 1000} seconds and ${Date.now()} milliseconds.`);

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

// Function to send settings to the backend
async function sendSettings(settings) {
    const statusMessage = document.getElementById('statusMessage');
    try {
        const response = await fetch('http://20.172.32.213:8080/settings', { // Adjust the URL if needed
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Settings updated:', data);
            statusMessage.textContent = 'Settings updated successfully.';
            statusMessage.style.color = 'green';
        } else {
            const errorData = await response.json();
            console.error('Failed to update settings:', errorData);
            statusMessage.textContent = 'Failed to update settings.';
            statusMessage.style.color = 'red';
        }
    } catch (error) {
        console.error('Error sending settings:', error);
        statusMessage.textContent = 'Error sending settings.';
        statusMessage.style.color = 'red';
    }

    // Clear the status message after 5 seconds
    setTimeout(() => {
        statusMessage.textContent = '';
    }, 5000);
}

// Function to send FPS settings to the backend (if needed in future)
// Currently, Target FPS is handled entirely on the client side
// If backend needs to know about FPS, implement here

// Function to handle settings form submission
function handleSettingsForm() {
    const settingsForm = document.getElementById('settingsForm');

    settingsForm.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent the default form submission

        // Gather the form data
        const prompt = document.getElementById('prompt').value.trim();
        const seedValue = document.getElementById('seed').value.trim();
        const inferenceStepsValue = document.getElementById('inference_steps').value.trim();
        const noiseStrengthValue = document.getElementById('noise_strength').value.trim();
        const conditioningScaleValue = document.getElementById('conditioning_scale').value.trim();

        // Prepare the settings object, only include fields that have been modified
        const settings = {};

        if (prompt !== "") {
            settings.prompt = prompt;
        }

        if (seedValue !== "") {
            const seed = parseInt(seedValue, 10);
            if (!isNaN(seed)) {
                settings.seed = seed;
            } else {
                alert('Seed must be an integer.');
                return;
            }
        }

        if (inferenceStepsValue !== "") {
            const inference_steps = parseInt(inferenceStepsValue, 10);
            if (!isNaN(inference_steps) && inference_steps >= 2) { // Ensure inference_steps >= 2
                settings.inference_steps = inference_steps;
            } else {
                alert('Inference Steps must be an integer greater than or equal to 2.');
                return;
            }
        }

        if (noiseStrengthValue !== "") {
            const noise_strength = parseFloat(noiseStrengthValue);
            if (!isNaN(noise_strength) && noise_strength >= 0.5 && noise_strength <= 5) { // Ensure 0.5 <= noise_strength <= 1
                settings.noise_strength = noise_strength;
            } else {
                alert('Noise Strength must be a number between 0.5 and 1.0.');
                return;
            }
        }

        if (conditioningScaleValue !== "") {
            const conditioning_scale = parseFloat(conditioningScaleValue);
            if (!isNaN(conditioning_scale) && conditioning_scale >= 0 && conditioning_scale <= 5) {
                settings.conditioning_scale = conditioning_scale;
            } else {
                alert('Conditioning Scale must be a number between 0.0 and 1.0.');
                return;
            }
        }

        // Send the settings to the backend
        sendSettings(settings);
    });
}

// Function to handle FPS form submission
function handleFPSForm() {
    const fpsForm = document.getElementById('fpsForm');

    fpsForm.addEventListener('submit', (event) => {
        event.preventDefault(); // Prevent the default form submission

        // Gather the form data
        const targetFPSValue = document.getElementById('target_fps').value.trim();

        // Prepare the FPS settings
        if (targetFPSValue !== "") {
            const target_fps = parseInt(targetFPSValue, 10);
            if (!isNaN(target_fps) && target_fps >= 1) {
                window.currentTargetFPS = target_fps; // Update the global FPS variable
                console.log(`Target FPS updated to: ${window.currentTargetFPS}`);

                // Optionally, display a status message
                const fpsStatusMessage = document.getElementById('fpsStatusMessage');
                fpsStatusMessage.textContent = 'Target FPS updated successfully.';
                fpsStatusMessage.style.color = 'green';

                // Clear the status message after 5 seconds
                setTimeout(() => {
                    fpsStatusMessage.textContent = '';
                }, 5000);
            } else {
                alert('Target FPS must be an integer greater than or equal to 1.');
                return;
            }
        }
    });
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

// Initialize the settings form handler
handleSettingsForm();

// Initialize the FPS form handler
handleFPSForm();

// Start the camera stream and WebSocket connection when the page loads
window.onload = async () => {
    await populateVideoSources();
    await startCameraStream();
};
