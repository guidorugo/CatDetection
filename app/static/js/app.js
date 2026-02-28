// Auth helper
function getToken() {
    return localStorage.getItem('access_token');
}

async function apiFetch(url, options = {}) {
    const token = getToken();
    if (!token && !url.includes('/auth/')) {
        window.location.href = '/login';
        return;
    }

    const headers = { 'Content-Type': 'application/json', ...options.headers };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch(url, { ...options, headers });

    if (resp.status === 401) {
        // Try refresh
        const refreshed = await refreshToken();
        if (refreshed) {
            headers['Authorization'] = `Bearer ${getToken()}`;
            const retry = await fetch(url, { ...options, headers });
            if (retry.ok) {
                if (retry.status === 204) return null;
                return retry.json();
            }
        }
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return;
    }

    if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${resp.status}`);
    }

    if (resp.status === 204) return null;
    return resp.json();
}

async function refreshToken() {
    const refresh = localStorage.getItem('refresh_token');
    if (!refresh) return false;

    try {
        const resp = await fetch('/api/v1/auth/refresh', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refresh }),
        });
        if (resp.ok) {
            const data = await resp.json();
            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('refresh_token', data.refresh_token);
            return true;
        }
    } catch (e) { /* ignore */ }
    return false;
}

function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    window.location.href = '/login';
}

function formatTime(isoString) {
    return new Date(isoString).toLocaleString();
}
