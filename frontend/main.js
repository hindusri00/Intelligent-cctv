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

// ---------- Confidence formatting ----------
// Used anywhere a fused identity confidence (face + body + attribute
// fusion - see modules/fusion_module.py) needs to show up in the UI as a
// small color-coded readout instead of a bare number.
function confidenceClass(value) {
    if (value === null || value === undefined) return '';
    if (value >= 0.7) return 'conf-high';
    if (value >= 0.5) return 'conf-mid';
    return 'conf-low';
}

function confidenceBadge(confidence, face, body, attribute) {
    if (confidence === null || confidence === undefined) return '';
    const parts = [];
    if (face !== null && face !== undefined) parts.push(`F ${Math.round(face * 100)}%`);
    if (body !== null && body !== undefined) parts.push(`B ${Math.round(body * 100)}%`);
    if (attribute !== null && attribute !== undefined) parts.push(`A ${Math.round(attribute * 100)}%`);
    const breakdown = parts.length ? ` <span class="conf-breakdown">(${parts.join(' \u00b7 ')})</span>` : '';
    return `<span class="conf-badge ${confidenceClass(confidence)}">${Math.round(confidence * 100)}%</span>${breakdown}`;
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

function setFeedMeta(sourceLabel, modeLabel) {
    document.getElementById('feedSourceLabel').textContent = sourceLabel;
    document.getElementById('feedModeLabel').textContent = modeLabel;
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
            pollCaseStatus();
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

// ---------- Case lifecycle (target-lock investigation) ----------
function fmtDateTime(unixSeconds) {
    if (!unixSeconds) return '—';
    return new Date(unixSeconds * 1000).toLocaleString();
}

async function pollCaseStatus() {
    const closeCaseBtn = document.getElementById('closeCaseBtn');
    const caseStatusLine = document.getElementById('caseStatusLine');
    try {
        const res = await fetch(`${API_BASE}/case/status`);
        const data = await res.json();
        setConn(true);

        if (data.open) {
            closeCaseBtn.style.display = 'block';
            caseStatusLine.style.display = 'block';
            caseStatusLine.textContent = `Case ${data.case_id} open since ${fmtDateTime(data.opened_at)}`;
        } else {
            closeCaseBtn.style.display = 'none';
            caseStatusLine.style.display = 'none';
        }
    } catch (error) {
        setConn(false);
    }
}

document.getElementById('closeCaseBtn').addEventListener('click', async () => {
    const confirmed = window.confirm(
        'Close this case now?\n\nThis generates the official case report (target attributes, ' +
        'sighting timeline with saved evidence frames, and every video/camera reviewed), then ' +
        'clears the target lock and tracking cache so a new case can start.'
    );
    if (!confirmed) return;

    const closeCaseBtn = document.getElementById('closeCaseBtn');
    closeCaseBtn.disabled = true;
    closeCaseBtn.textContent = 'Closing case\u2026';
    showToast('Generating case report\u2026');

    try {
        const res = await fetch(`${API_BASE}/case/close`, { method: 'POST' });
        const data = await res.json();
        setConn(true);

        if (data.status === 'closed') {
            showToast(`Case ${data.case.case_id} closed \u2014 report ready.`);
            resetCaseUI();
            await pollCaseStatus();
            await loadReports();
        } else {
            showToast(data.error || 'Could not close the case.');
        }
    } catch (error) {
        showToast('Server connection failed.');
        setConn(false);
    } finally {
        closeCaseBtn.disabled = false;
        closeCaseBtn.textContent = 'Close case & generate report';
    }
});

// Clears every bit of UI state tied to the just-closed case so the console
// reads as genuinely fresh for the next investigation, matching the backend
// reset (trackers/target/camera cache) that /api/case/close performs.
function resetCaseUI() {
    targetThumb.innerHTML = 'NO IMG';
    targetImageInput.value = '';
    const status = document.getElementById('targetStatus');
    status.className = '';
    status.textContent = 'No target selected';
    document.getElementById('targetConfidence').innerHTML = '';
    document.getElementById('statTarget').textContent = 'none';
    document.getElementById('statMatches').textContent = '0';
}

async function loadReports() {
    const reportsBox = document.getElementById('reportsBox');
    try {
        const res = await fetch(`${API_BASE}/cases`);
        const data = await res.json();
        setConn(true);

        const closed = data.cases.filter(c => c.status === 'closed');

        if (closed.length === 0) {
            reportsBox.innerHTML = '<p class="empty-note">No cases closed yet.</p>';
            return;
        }

        reportsBox.innerHTML = closed.map(c => `
            <div class="report-card">
                <div class="row1"><span>#${c.case_id}</span><span>${c.target_person_id}</span></div>
                <div class="row2">${fmtDateTime(c.opened_at)} &rarr; ${fmtDateTime(c.closed_at)}</div>
                ${c.report_pdf_path
                    ? `<button class="btn-primary dl-btn" data-case-id="${c.case_id}">Download official report (PDF)</button>`
                    : `<span class="empty-note">No report on file</span>`}
            </div>`).join('');
    } catch (error) {
        setConn(false);
    }
}

document.getElementById('reportsBox').addEventListener('click', (e) => {
    const btn = e.target.closest('.dl-btn');
    if (!btn) return;
    window.open(`${API_BASE}/cases/${btn.dataset.caseId}/report`, '_blank');
});

pollCaseStatus();
setInterval(pollCaseStatus, 4000);
loadReports();
setInterval(loadReports, 15000);

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
            setFeedMeta(videoInput.files[0].name, 'recorded footage');
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
    setFeedMeta('local camera', 'live');
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
            const conf = item.confidence !== undefined && item.confidence !== null
                ? `<span class="conf-badge ${confidenceClass(item.confidence)}">${Math.round(item.confidence * 100)}%</span>`
                : '';
            html += `
                <div class="result-card">
                    <div class="row1"><span class="pid">#${item.person_id}</span><span>${conf} ${item.timestamp}</span></div>
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

        // Show the fused-confidence breakdown for the most recent target
        // sighting next to the lock status, so "locked" isn't just a
        // binary - you can see how confident that lock actually is.
        const targetConfidenceEl = document.getElementById('targetConfidence');
        if (sightings.length > 0 && sightings[0].confidence !== null && sightings[0].confidence !== undefined) {
            targetConfidenceEl.innerHTML = `Last match: ${confidenceBadge(sightings[0].confidence, sightings[0].face_similarity, sightings[0].body_similarity, null)}`;
        } else {
            targetConfidenceEl.innerHTML = '';
        }

        if (data.alerts.length > 0) {
            alertBox.innerHTML = '';
            data.alerts.forEach(alert => {
                // A pure "Target Located" sighting is informational (lock
                // confirmed), not a rule violation - style it teal instead
                // of red so it doesn't read as a threat.
                const isPureSighting = alert.reason === 'Target Located';
                const conf = confidenceBadge(alert.confidence, alert.face_similarity, alert.body_similarity, null);
                alertBox.innerHTML += `
                    <div class="alert-card${isPureSighting ? ' alert-sighting' : ''}">
                        <div class="row1"><button class="ts-btn" data-ts="${alert.timestamp}">${isPureSighting ? 'SIGHTING' : 'ALERT'} &middot; ${alert.timestamp}</button><span>track #${alert.track_id}</span></div>
                        <div class="reason">${alert.reason}${conf ? ` &middot; ${conf}` : ''}</div>
                        <div class="who">person ID: ${alert.person_id}</div>
                    </div>`;
            });
        } else {
            alertBox.innerHTML = '<p class="no-alerts">No threats matching active rules.</p>';
        }

        renderLog(data.alerts);
    } catch (error) {
        setConn(false);
    }
}

// ---------- Event log (main-panel table) ----------
function renderLog(alerts) {
    const body = document.getElementById('logTableBody');
    const count = document.getElementById('logCount');
    count.textContent = `${alerts.length} event${alerts.length === 1 ? '' : 's'}`;

    if (!alerts.length) {
        body.innerHTML = `<tr class="log-empty-row"><td colspan="5">No events recorded yet &mdash; load footage or go live to begin logging.</td></tr>`;
        return;
    }

    // Newest activity first, same source data as the Threat Alerts panel,
    // just laid out as a scannable table for the main screen.
    const rows = [...alerts].reverse();
    body.innerHTML = rows.map(alert => {
        const isPureSighting = alert.reason === 'Target Located';
        const conf = alert.confidence !== null && alert.confidence !== undefined
            ? confidenceBadge(alert.confidence, alert.face_similarity, alert.body_similarity, null)
            : '&mdash;';
        return `
            <tr class="${isPureSighting ? 'row-sighting' : 'row-alert'}">
                <td class="col-time">${alert.timestamp}</td>
                <td class="col-track">#${alert.track_id}</td>
                <td class="col-person">${alert.person_id}</td>
                <td class="col-event">${alert.reason}</td>
                <td class="col-conf">${conf}</td>
            </tr>`;
    }).join('');
}

pollAlerts();
setInterval(pollAlerts, 2500);

// ---------- Poll identity roster ----------
function timeAgo(unixSeconds) {
    if (!unixSeconds) return '—';
    const diff = Date.now() / 1000 - unixSeconds;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

async function pollRoster() {
    const rosterBox = document.getElementById('rosterBox');
    try {
        const res = await fetch(`${API_BASE}/identities`);
        const data = await res.json();
        setConn(true);

        document.getElementById('statIdentities').textContent = data.identities.length;

        if (data.identities.length === 0) {
            rosterBox.innerHTML = '<p class="empty-note">No identities recorded yet.</p>';
            return;
        }

        let html = '';
        data.identities.slice(0, 25).forEach(identity => {
            const badges = [
                identity.has_face ? '<span class="mini-badge face">FACE</span>' : '',
                identity.has_body ? '<span class="mini-badge body">BODY</span>' : '',
            ].join('');
            html += `
                <div class="roster-row">
                    <span class="pid">#${identity.person_id}</span>
                    ${badges}
                    <span class="roster-meta">${identity.observations} sightings &middot; last seen ${timeAgo(identity.last_seen)}</span>
                </div>`;
        });
        rosterBox.innerHTML = html;
    } catch (error) {
        setConn(false);
    }
}

pollRoster();
setInterval(pollRoster, 5000);

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
