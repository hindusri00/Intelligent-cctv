const API_BASE = "http://127.0.0.1:5000/api";

// ---------- small helpers ----------
function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.remove('show'), 2600);
}

function setConn(ok) {
    const dot = document.getElementById('connDot');
    const label = document.getElementById('connLabel');
    if (ok) {
        dot.classList.add('live');
        label.textContent = 'linked to backend';
    } else {
        dot.classList.remove('live');
        label.textContent = 'backend unreachable';
    }
}

function tickClock() {
    const el = document.getElementById('clock');
    if (el) el.textContent = new Date().toLocaleTimeString();
}
tickClock();
setInterval(tickClock, 1000);

// ---------- Stream state ----------
let currentVideoPath = null;   // uploaded footage path, if any
let isLive = false;            // true once the live camera has been requested

const videoFrame = document.getElementById('videoFrame');
const streamViewer = document.getElementById('streamViewer');

function setFrameState(state) {
    // state: 'empty' | 'loading' | 'active'
    videoFrame.classList.toggle('is-empty', state === 'empty');
    videoFrame.classList.toggle('is-loading', state === 'loading');
}

function startStream(url, loadingLabel) {
    document.getElementById('loadingLabel').textContent = loadingLabel || 'Connecting to feed\u2026';
    setFrameState('loading');
    streamViewer.src = url;
}

// MJPEG streams fire the <img> 'load' event on every new frame, so the
// first one is a reliable signal that the feed is actually flowing.
streamViewer.addEventListener('load', () => setFrameState('active'));
streamViewer.addEventListener('error', () => {
    setFrameState('empty');
    showToast('Could not connect to the feed.');
    setConn(false);
});

// ---------- Target person selection ----------
const targetImageInput = document.getElementById('targetImageInput');
const targetThumb = document.getElementById('targetThumb');

targetImageInput.addEventListener('change', () => {
    const file = targetImageInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        targetThumb.innerHTML = `<img src="${e.target.result}" alt="Target reference">`;
    };
    reader.readAsDataURL(file);
});

document.getElementById('targetUploadBtn').addEventListener('click', async () => {
    const status = document.getElementById('targetStatus');

    if (!targetImageInput.files[0]) {
        status.className = 'error';
        status.textContent = 'Please select a target image first.';
        return;
    }

    const formData = new FormData();
    formData.append('image', targetImageInput.files[0]);

    status.className = '';
    status.textContent = 'Setting target...';

    try {
        const res = await fetch(`${API_BASE}/target`, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            status.className = 'locked';
            status.textContent = `Target locked: ${data.target_id}`;
            document.getElementById('statTarget').textContent = data.target_id;
            setConn(true);
        } else {
            status.className = 'error';
            status.textContent = data.error || 'Failed to set target.';
        }
    } catch (error) {
        status.className = 'error';
        status.textContent = 'Server connection failed.';
        setConn(false);
    }
});

// ---------- Upload / load footage ----------
const videoInput = document.getElementById('videoInput');
const videoFileName = document.getElementById('videoFileName');

videoInput.addEventListener('change', () => {
    videoFileName.textContent = videoInput.files[0] ? videoInput.files[0].name : 'no file selected';
});

document.getElementById('uploadBtn').addEventListener('click', async () => {
    if (!videoInput.files[0]) {
        showToast('Select a video file first.');
        return;
    }

    const formData = new FormData();
    formData.append('video', videoInput.files[0]);

    showToast('Uploading footage...');

    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status === 'success') {
            currentVideoPath = data.video_path;
            isLive = false;
            startStream(
                `${API_BASE}/stream?path=${encodeURIComponent(data.video_path)}&_=${Date.now()}`,
                'Loading footage\u2026'
            );
            showToast('Stream loaded.');
            setConn(true);
        } else {
            showToast(data.error || 'Upload failed.');
        }
    } catch (error) {
        showToast('Server connection failed.');
        setConn(false);
    }
});

// ---------- Live camera ----------
document.getElementById('liveBtn').addEventListener('click', () => {
    currentVideoPath = null;
    isLive = true;
    // Runs the SERVER machine's camera (this Flask process), not the
    // viewer's browser camera - fine for a locally-run console like this.
    startStream(`${API_BASE}/stream?_=${Date.now()}`, 'Starting live camera\u2026');
    showToast('Switching to live camera.');
});

// ---------- Attribute search ----------
document.getElementById('searchBtn').addEventListener('click', runSearch);
document.getElementById('searchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runSearch();
});

async function runSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    const resultsContainer = document.getElementById('searchResults');
    resultsContainer.innerHTML = `<div class="results-meta">Searching&hellip;</div>`;

    try {
        const res = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        setConn(true);

        if (data.results.length === 0) {
            resultsContainer.innerHTML =
                `<div class="results-meta">Matches for "${data.query}"</div>
                 <p class="empty-note">No matching individuals found.</p>`;
            return;
        }

        let html = `<div class="results-meta">Matches for "${data.query}" &middot; ${data.results.length}</div>`;
        data.results.forEach(item => {
            html += `
                <div class="result-card">
                    <div class="row1"><span class="pid">#${item.person_id}</span><span>${item.timestamp}</span></div>
                    <div class="attrs">
                        <span>shirt</span> ${item.shirt || '—'} &nbsp;
                        <span>pant</span> ${item.pant || '—'} &nbsp;
                        <span>hair</span> ${item.hair || '—'} &nbsp;
                        <span>mask</span> ${item.mask || '—'}
                    </div>
                </div>`;
        });
        resultsContainer.innerHTML = html;
    } catch (error) {
        resultsContainer.innerHTML = `<p class="empty-note">Server connection failed.</p>`;
        setConn(false);
    }
}

// ---------- Blacklist rules ----------
document.getElementById('updateRulesBtn').addEventListener('click', async () => {
    const blacklist = {
        mask: document.getElementById('chkMask').checked,
        tattoo: document.getElementById('chkTattoo').checked,
        bag: document.getElementById('chkBag').checked,
        spectacles: document.getElementById('chkGlasses').checked
    };

    try {
        await fetch(`${API_BASE}/blacklist`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ blacklist })
        });
        showToast('Alert rules updated.');
        setConn(true);
    } catch (error) {
        showToast('Server connection failed.');
        setConn(false);
    }
});

// ---------- Poll alerts ----------
async function pollAlerts() {
    const alertBox = document.getElementById('alertBox');
    try {
        const res = await fetch(`${API_BASE}/alerts`);
        const data = await res.json();
        setConn(true);

        const sightings = data.alerts.filter(a => a.reason.includes('Target Located'));
        document.getElementById('statMatches').textContent = sightings.length;
        document.getElementById('statAlerts').textContent = data.alerts.length;

        if (data.alerts.length > 0) {
            alertBox.innerHTML = '';
            data.alerts.forEach(alert => {
                // A pure "Target Located" sighting is informational (lock
                // confirmed), not a rule violation - style it teal instead
                // of red so it doesn't read as a threat.
                const isPureSighting = alert.reason === 'Target Located';
                alertBox.innerHTML += `
                    <div class="alert-card${isPureSighting ? ' alert-sighting' : ''}">
                        <div class="row1"><button class="ts-btn" data-ts="${alert.timestamp}">${isPureSighting ? 'SIGHTING' : 'ALERT'} &middot; ${alert.timestamp}</button><span>track #${alert.track_id}</span></div>
                        <div class="reason">${alert.reason}</div>
                        <div class="who">person ID: ${alert.person_id}</div>
                    </div>`;
            });
        } else {
            alertBox.innerHTML = '<p class="no-alerts">No threats matching active rules.</p>';
        }
    } catch (error) {
        setConn(false);
    }
}

pollAlerts();
setInterval(pollAlerts, 2500);

// ---------- Click an alert's timestamp to rewind the feed there ----------
document.getElementById('alertBox').addEventListener('click', (e) => {
    const btn = e.target.closest('.ts-btn');
    if (!btn) return;

    if (isLive || !currentVideoPath) {
        showToast('Rewind only works on loaded footage, not the live camera.');
        return;
    }

    const seconds = parseFloat(btn.dataset.ts); // "12.34s" -> 12.34
    if (Number.isNaN(seconds)) return;

    startStream(
        `${API_BASE}/stream?path=${encodeURIComponent(currentVideoPath)}&start=${seconds}&_=${Date.now()}`,
        `Rewinding to ${btn.dataset.ts}\u2026`
    );
});
