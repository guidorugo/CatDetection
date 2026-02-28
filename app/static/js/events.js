// Event feed rendering and real-time updates

function renderEventList(events, container) {
    if (!events || events.length === 0) {
        container.innerHTML = '<p class="empty-state">No events found.</p>';
        return;
    }

    container.innerHTML = events.map(evt => `
        <div class="event-item">
            ${evt.thumbnail_path ?
                `<img class="event-thumb" src="/api/v1/events/${evt.id}/thumbnail" alt="Detection">` :
                `<div class="event-thumb"></div>`}
            <div class="event-info">
                <h4>${evt.cat_name || 'Unknown Cat'}</h4>
                <p>Camera: ${evt.camera_name || 'Camera ' + evt.camera_id} | Confidence: ${(evt.detection_confidence * 100).toFixed(0)}%${evt.identification_confidence ? ' | ID: ' + (evt.identification_confidence * 100).toFixed(0) + '%' : ''}</p>
            </div>
            <div class="event-meta">
                <div>${formatTime(evt.timestamp)}</div>
                ${evt.recording_id ? `<a href="/recordings?id=${evt.recording_id}" class="btn btn-sm" style="margin-top:4px">Recording</a>` : ''}
            </div>
        </div>
    `).join('');
}

let eventWs = null;

function connectEventStream() {
    const token = getToken();
    if (!token) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/events`;

    eventWs = new WebSocket(wsUrl);

    eventWs.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'detection') {
                // Prepend new event to the recent events list
                const list = document.getElementById('recentEvents');
                if (list) {
                    const item = document.createElement('div');
                    item.className = 'event-item';
                    item.innerHTML = `
                        <div class="event-thumb"></div>
                        <div class="event-info">
                            <h4>${data.cat_name || 'Unknown Cat'}</h4>
                            <p>Camera ${data.camera_id} | Confidence: ${(data.confidence * 100).toFixed(0)}%</p>
                        </div>
                        <div class="event-meta">
                            <div>${new Date().toLocaleString()}</div>
                        </div>
                    `;
                    list.prepend(item);
                    // Keep max 20 items
                    while (list.children.length > 20) {
                        list.removeChild(list.lastChild);
                    }
                }
            }
        } catch (e) {
            console.error('Event parse error:', e);
        }
    };

    eventWs.onclose = () => {
        setTimeout(connectEventStream, 3000);
    };
}
