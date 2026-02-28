// WebSocket live stream handler

const liveStreams = {};

function connectLiveStream(cameraId) {
    const token = getToken();
    if (!token) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/live/${cameraId}`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    const canvas = document.getElementById(`canvas-${cameraId}`);
    const noFeed = document.getElementById(`nofeed-${cameraId}`);
    const fpsDisplay = document.getElementById(`fps-${cameraId}`);

    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let frameCount = 0;
    let lastFpsUpdate = Date.now();

    ws.onopen = () => {
        console.log(`Live stream connected: camera ${cameraId}`);
        if (noFeed) noFeed.style.display = 'none';
    };

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);

                frameCount++;
                const now = Date.now();
                if (now - lastFpsUpdate >= 1000) {
                    if (fpsDisplay) {
                        fpsDisplay.textContent = `${frameCount} FPS`;
                    }
                    frameCount = 0;
                    lastFpsUpdate = now;
                }
            };
            img.src = url;
        }
    };

    ws.onclose = () => {
        console.log(`Live stream disconnected: camera ${cameraId}`);
        if (noFeed) noFeed.style.display = 'flex';
        // Reconnect after delay
        setTimeout(() => connectLiveStream(cameraId), 3000);
    };

    ws.onerror = (err) => {
        console.error(`Live stream error: camera ${cameraId}`, err);
    };

    liveStreams[cameraId] = ws;
}

function disconnectLiveStream(cameraId) {
    const ws = liveStreams[cameraId];
    if (ws) {
        ws.close();
        delete liveStreams[cameraId];
    }
}
