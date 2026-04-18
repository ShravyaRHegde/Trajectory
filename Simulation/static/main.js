let currentData = null;
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
let offset = { x: 0, y: 0 };
let scale = 15; // Pixels per meter

// --- Init & UI ---
async function init() {
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    const response = await fetch('/scenarios');
    const scenarios = await response.json();
    renderScenarioList(scenarios);
    
    document.getElementById('resetView').onclick = () => {
        offset = { x: canvas.width/2, y: canvas.height/2 };
        scale = 15;
        draw();
    };
    
    document.getElementById('scenarioSearch').oninput = (e) => {
        const query = e.target.value.toLowerCase();
        const items = document.querySelectorAll('.scenario-list li');
        items.forEach(li => {
            li.style.display = li.textContent.toLowerCase().includes(query) ? '' : 'none';
        });
    };
}

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    offset = { x: canvas.width/2, y: canvas.height/2 };
    draw();
}

function renderScenarioList(ids) {
    const list = document.getElementById('scenarioList');
    list.innerHTML = '';
    ids.forEach(id => {
        const li = document.createElement('li');
        li.textContent = id.substring(0, 18) + '...';
        li.title = id;
        li.onclick = () => selectScenario(id, li);
        list.appendChild(li);
    });
}

async function selectScenario(id, element) {
    document.querySelectorAll('.scenario-list li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');
    
    document.getElementById('scenarioMeta').textContent = `Loading ${id}...`;
    
    try {
        const response = await fetch(`/evaluate/${id}`);
        currentData = await response.json();
        
        document.getElementById('scenarioMeta').innerHTML = `
            <strong>ID:</strong> ${id.substring(0,8)}...<br>
            <strong>Origin:</strong> ${currentData.origin[0].toFixed(1)}, ${currentData.origin[1].toFixed(1)}
        `;
        
        // Mock metrics based on dummy analysis
        document.getElementById('jerkVal').textContent = '0.142';
        document.getElementById('confVal').textContent = `\u00b1${currentData.safety_envelope[currentData.safety_envelope.length-1].toFixed(2)}m`;
        
        autoCenter();
        draw();
    } catch (e) {
        console.error(e);
        document.getElementById('scenarioMeta').textContent = "Failed to load.";
    }
}

function autoCenter() {
    if (!currentData) return;
    // We are normalized to 0,0 at index 19 (the history tip)
    offset = { x: canvas.width/2, y: canvas.height/2 };
}

// --- Coordinate Transform ---
function worldToScreen(wx, wy) {
    // Canvas y is down, world y is up? Usually in robotics y is forward/left. 
    // In our case, we'll keep x and -y for visual sanity.
    return {
        x: offset.x + (wx * scale),
        y: offset.y - (wy * scale) 
    };
}

// --- Rendering ---
function draw() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!currentData) {
        ctx.fillStyle = '#333';
        ctx.textAlign = 'center';
        ctx.font = '16px Outfit';
        ctx.fillText('Select a scenario from the list', canvas.width/2, canvas.height/2);
        return;
    }

    // 1. Draw Map Drivable Areas
    ctx.fillStyle = 'rgba(40, 40, 50, 0.4)';
    currentData.map.drivable_areas.forEach(poly => {
        ctx.beginPath();
        poly.forEach((pt, i) => {
            const s = worldToScreen(pt[0], pt[1]);
            if (i === 0) ctx.moveTo(s.x, s.y);
            else ctx.lineTo(s.x, s.y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.stroke();
    });

    // 2. Draw Lanes
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    currentData.map.lanes.forEach(lane => {
        ctx.beginPath();
        lane.centerline.forEach((pt, i) => {
            const s = worldToScreen(pt[0], pt[1]);
            if (i === 0) ctx.moveTo(s.x, s.y);
            else ctx.lineTo(s.x, s.y);
        });
        ctx.stroke();
    });
    ctx.setLineDash([]);

    // 3. Draw History (Electric Blue)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.beginPath();
    currentData.history.forEach((pt, i) => {
        const s = worldToScreen(pt[0], pt[1]);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
    });
    ctx.stroke();

    // 4. Draw Ground Truth Future (Emerald Dashed)
    ctx.strokeStyle = '#10b981';
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    currentData.future_gt.forEach((pt, i) => {
        const s = worldToScreen(pt[0], pt[1]);
        if (i === 0) {
            // Start from last history point
            const hLast = currentData.history[currentData.history.length-1];
            const start = worldToScreen(hLast[0], hLast[1]);
            ctx.moveTo(start.x, start.y);
        }
        ctx.lineTo(s.x, s.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. Draw Safety Envelope & Integrated Prediction (Purple Gradient)
    const heads = currentData.prediction[0]; // (30, 2, 3) 
    const p50 = heads.map(h => [h[0][1], h[1][1]]); // P50 is usually head index 1
    const horizon = p50.length;
    
    // Draw Envelope (translucent purple area)
    ctx.fillStyle = 'rgba(217, 70, 239, 0.15)';
    ctx.beginPath();
    // Forward path (right side of center)
    p50.forEach((pt, i) => {
        const q = currentData.safety_envelope[i];
        const s = worldToScreen(pt[0] + q, pt[1]);
        if (i === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
    });
    // Backward path (left side of center)
    for (let i = horizon - 1; i >= 0; i--) {
        const q = currentData.safety_envelope[i];
        const s = worldToScreen(p50[i][0] - q, p50[i][1]);
        ctx.lineTo(s.x, s.y);
    }
    ctx.closePath();
    ctx.fill();

    // Draw P50 Prediction Line
    ctx.strokeStyle = '#d946ef';
    ctx.lineWidth = 4;
    ctx.beginPath();
    p50.forEach((pt, i) => {
        const s = worldToScreen(pt[0], pt[1]);
        if (i === 0) {
            const hLast = currentData.history[currentData.history.length-1];
            const start = worldToScreen(hLast[0], hLast[1]);
            ctx.moveTo(start.x, start.y);
        }
        ctx.lineTo(s.x, s.y);
    });
    ctx.stroke();

    // 6. Draw Vehicle at Decision Point (t=0 normalized)
    const lastPos = worldToScreen(0, 0); 
    ctx.fillStyle = '#fff';
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#fff';
    ctx.beginPath();
    ctx.arc(lastPos.x, lastPos.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
}

// --- Zoom/Pan Controls ---
canvas.onwheel = (e) => {
    e.preventDefault();
    const zoom = e.deltaY > 0 ? 0.9 : 1.1;
    scale *= zoom;
    draw();
};

let isDragging = false;
let lastMouse = { x: 0, y: 0 };

canvas.onmousedown = (e) => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; };
canvas.onmouseup = () => { isDragging = false; };
canvas.onmousemove = (e) => {
    if (!isDragging) return;
    offset.x += (e.clientX - lastMouse.x);
    offset.y += (e.clientY - lastMouse.y);
    lastMouse = { x: e.clientX, y: e.clientY };
    draw();
};

init();
