// Image Generation Studio - Frontend JavaScript

// Global variables
let ws = null;
let currentImage = null;
let generationHistory = [];
let isGenerating = false;
let generationStartTime = null;
let progressInterval = null;

// Sample prompts for random selection
const samplePrompts = [
    "A majestic dragon soaring through cloudy skies, fantasy art style",
    "Cyberpunk city at night with neon lights and flying cars",
    "Serene Japanese garden with cherry blossoms and koi pond",
    "Astronaut exploring an alien planet with bioluminescent plants",
    "Steampunk laboratory with brass machinery and glowing tubes",
    "Northern lights over a snowy mountain landscape",
    "Underwater coral reef teeming with colorful fish",
    "Ancient library with floating books and magical atmosphere",
    "Robot chef cooking in a futuristic kitchen",
    "Enchanted forest path with glowing mushrooms and fireflies"
];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeWebSocket();
    loadHistory();
    setupEventListeners();
    updateDimensionInfo();
    updateEstimatedTime();
});

// WebSocket connection
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };
    
    ws.onclose = function() {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        // Attempt to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
}

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connectionStatus');
    if (connected) {
        statusEl.className = 'badge bg-success me-3';
        statusEl.innerHTML = '<i class="bi bi-wifi"></i> Connected';
    } else {
        statusEl.className = 'badge bg-secondary me-3';
        statusEl.innerHTML = '<i class="bi bi-wifi-off"></i> Disconnected';
    }
}

function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'queue_update':
            // Could show queue position if needed
            break;
        case 'generation_complete':
            // Generation completed successfully
            break;
        case 'generation_error':
            if (data.request_id && isGenerating) {
                showError('Generation failed: ' + data.error);
                stopGeneration();
            }
            break;
    }
}

// Event listeners
function setupEventListeners() {
    // Form submission
    document.getElementById('generationForm').addEventListener('submit', handleGenerate);
    
    // Dimension inputs
    document.getElementById('width').addEventListener('input', updateDimensionInfo);
    document.getElementById('height').addEventListener('input', updateDimensionInfo);
    document.getElementById('width').addEventListener('input', updateEstimatedTime);
    document.getElementById('height').addEventListener('input', updateEstimatedTime);
    
    // Steps slider
    document.getElementById('steps').addEventListener('input', function() {
        document.getElementById('stepsValue').textContent = this.value;
        updateEstimatedTime();
    });
    
    // Guidance slider
    document.getElementById('guidance').addEventListener('input', function() {
        document.getElementById('guidanceValue').textContent = this.value;
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to generate
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (!isGenerating) {
                document.getElementById('generateBtn').click();
            }
        }
    });
}

// Form handling
async function handleGenerate(e) {
    e.preventDefault();
    
    if (isGenerating) return;
    
    const formData = new FormData(e.target);
    const data = {
        prompt: formData.get('prompt'),
        width: parseInt(formData.get('width')),
        height: parseInt(formData.get('height')),
        steps: parseInt(formData.get('steps')),
        guidance: parseFloat(formData.get('guidance'))
    };
    
    // Add seed if provided
    const seed = formData.get('seed');
    if (seed && seed.trim() !== '') {
        data.seed = parseInt(seed);
    }
    
    // Validate dimensions
    if (data.width % 16 !== 0 || data.height % 16 !== 0) {
        showError('Width and height must be divisible by 16');
        return;
    }
    
    const totalPixels = data.width * data.height;
    if (totalPixels > 8388608) {
        showError('Total pixels exceed maximum limit (8,388,608)');
        return;
    }
    
    startGeneration();
    
    try {
        const response = await fetch('/v1/images/generations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }
        
        const result = await response.json();
        displayResult(result, data);
        addToHistory(result, data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        stopGeneration();
    }
}

function startGeneration() {
    isGenerating = true;
    generationStartTime = Date.now();
    
    // Update UI
    document.getElementById('generateBtn').classList.add('loading');
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('resultContainer').classList.add('d-none');
    document.getElementById('progressContainer').classList.remove('d-none');
    
    // Start progress timer
    progressInterval = setInterval(updateProgressTime, 100);
}

function stopGeneration() {
    isGenerating = false;
    
    // Update UI
    document.getElementById('generateBtn').classList.remove('loading');
    document.getElementById('generateBtn').disabled = false;
    document.getElementById('progressContainer').classList.add('d-none');
    
    // Stop progress timer
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

function updateProgressTime() {
    if (generationStartTime) {
        const elapsed = (Date.now() - generationStartTime) / 1000;
        document.getElementById('progressTime').textContent = elapsed.toFixed(1) + 's';
    }
}

// Display functions
function displayResult(result, params) {
    currentImage = result;
    
    const html = `
        <div class="card">
            <div class="card-body">
                <img src="data:image/png;base64,${result.image}" 
                     alt="Generated image" 
                     class="result-image mb-3"
                     onclick="showImageModal()">
                <div class="row">
                    <div class="col-md-6">
                        <p class="mb-1"><strong>Prompt:</strong> ${escapeHtml(params.prompt)}</p>
                        <p class="mb-1"><strong>Dimensions:</strong> ${params.width}×${params.height}</p>
                        <p class="mb-1"><strong>Steps:</strong> ${params.steps}</p>
                    </div>
                    <div class="col-md-6">
                        <p class="mb-1"><strong>Guidance:</strong> ${params.guidance}</p>
                        <p class="mb-1"><strong>Seed:</strong> ${result.seed || 'Random'}</p>
                        <p class="mb-1"><strong>Generation Time:</strong> ${result.gen_time.toFixed(2)}s</p>
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-primary btn-sm" onclick="downloadImage()">
                        <i class="bi bi-download"></i> Download
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="copySettings()">
                        <i class="bi bi-clipboard"></i> Copy Settings
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="useAsSeed()">
                        <i class="bi bi-dice-3"></i> Use Seed
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('resultContainer').innerHTML = html;
    document.getElementById('resultContainer').classList.remove('d-none');
}

// History management
function loadHistory() {
    const saved = localStorage.getItem('generationHistory');
    if (saved) {
        generationHistory = JSON.parse(saved);
        displayHistory();
    }
}

function saveHistory() {
    // Keep only last 20 items
    if (generationHistory.length > 20) {
        generationHistory = generationHistory.slice(-20);
    }
    localStorage.setItem('generationHistory', JSON.stringify(generationHistory));
}

function addToHistory(result, params) {
    // Create a thumbnail to save space in localStorage
    const thumbnail = createThumbnail(result.image, 128, 128);
    
    const historyItem = {
        id: Date.now(),
        thumbnail: thumbnail,  // Store thumbnail instead of full image
        fullImage: null,       // Don't store full image in history
        params: params,
        seed: result.seed,
        gen_time: result.gen_time,
        timestamp: new Date().toISOString()
    };
    
    generationHistory.push(historyItem);
    saveHistory();
    displayHistory();
}

function createThumbnail(base64Image, maxWidth, maxHeight) {
    // Create thumbnail on canvas to reduce size
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    // This is synchronous for now, could be made async if needed
    img.src = 'data:image/png;base64,' + base64Image;
    
    // Calculate thumbnail dimensions
    let width = img.width;
    let height = img.height;
    
    if (width > height) {
        if (width > maxWidth) {
            height = Math.round(height * maxWidth / width);
            width = maxWidth;
        }
    } else {
        if (height > maxHeight) {
            width = Math.round(width * maxHeight / height);
            height = maxHeight;
        }
    }
    
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);
    
    // Return base64 thumbnail without data URL prefix
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
}

function displayHistory() {
    const container = document.getElementById('historyContainer');
    
    if (generationHistory.length === 0) {
        container.innerHTML = '<p class="text-muted text-center p-3">No history yet</p>';
        return;
    }
    
    const html = generationHistory.slice().reverse().map(item => `
        <div class="history-item" onclick="loadFromHistory('${item.id}')" title="${escapeHtml(item.params.prompt)}">
            <img src="data:image/jpeg;base64,${item.thumbnail}" alt="History image">
            <div class="history-info">
                ${item.params.width}×${item.params.height}
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

function loadFromHistory(id) {
    const item = generationHistory.find(h => h.id == id);
    if (!item) return;
    
    // Load parameters
    document.getElementById('prompt').value = item.params.prompt;
    document.getElementById('width').value = item.params.width;
    document.getElementById('height').value = item.params.height;
    document.getElementById('steps').value = item.params.steps;
    document.getElementById('guidance').value = item.params.guidance;
    if (item.seed) {
        document.getElementById('seed').value = item.seed;
    }
    
    // Update displays
    document.getElementById('stepsValue').textContent = item.params.steps;
    document.getElementById('guidanceValue').textContent = item.params.guidance;
    updateDimensionInfo();
    updateEstimatedTime();
    
    // Note: We don't store full images in history anymore, just show params
    showSuccess('Settings loaded from history');
}

function clearHistory() {
    if (confirm('Clear all generation history?')) {
        generationHistory = [];
        saveHistory();
        displayHistory();
    }
}

// Utility functions
function setDimensions(width, height) {
    document.getElementById('width').value = width;
    document.getElementById('height').value = height;
    updateDimensionInfo();
    updateEstimatedTime();
}

function updateDimensionInfo() {
    const width = parseInt(document.getElementById('width').value);
    const height = parseInt(document.getElementById('height').value);
    
    // Update pixel count
    const pixels = width * height;
    document.getElementById('pixelCount').textContent = pixels.toLocaleString();
    
    // Update aspect ratio
    const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
    const divisor = gcd(width, height);
    const ratioW = width / divisor;
    const ratioH = height / divisor;
    document.getElementById('aspectRatio').textContent = `(${ratioW}:${ratioH})`;
}

function updateEstimatedTime() {
    const width = parseInt(document.getElementById('width').value);
    const height = parseInt(document.getElementById('height').value);
    const steps = parseInt(document.getElementById('steps').value);
    
    // Simple estimation formula (adjust based on your hardware)
    const pixels = width * height;
    const baseTime = 0.3; // Base time in seconds
    const pixelFactor = pixels / 1000000; // Per million pixels
    const stepFactor = steps / 4; // Normalized to 4 steps
    
    const estimated = baseTime + (pixelFactor * 0.2) + (stepFactor * 0.15);
    
    document.getElementById('estimatedTime').textContent = 
        `Estimated time: ~${estimated.toFixed(1)}s`;
}

function randomPrompt() {
    const prompt = samplePrompts[Math.floor(Math.random() * samplePrompts.length)];
    document.getElementById('prompt').value = prompt;
}

function clearPrompt() {
    document.getElementById('prompt').value = '';
    document.getElementById('prompt').focus();
}

function randomSeed() {
    const seed = Math.floor(Math.random() * 2147483647);
    document.getElementById('seed').value = seed;
}

function showImageModal() {
    if (!currentImage) return;
    
    document.getElementById('modalImage').src = 'data:image/png;base64,' + currentImage.image;
    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
}

function downloadImage() {
    if (!currentImage) return;
    
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + currentImage.image;
    link.download = `generated_${Date.now()}.png`;
    link.click();
}

function copySettings() {
    const settings = {
        prompt: document.getElementById('prompt').value,
        width: parseInt(document.getElementById('width').value),
        height: parseInt(document.getElementById('height').value),
        steps: parseInt(document.getElementById('steps').value),
        guidance: parseFloat(document.getElementById('guidance').value),
        seed: currentImage ? currentImage.seed : null
    };
    
    navigator.clipboard.writeText(JSON.stringify(settings, null, 2))
        .then(() => showSuccess('Settings copied to clipboard'))
        .catch(() => showError('Failed to copy settings'));
}

function useAsSeed() {
    if (currentImage && currentImage.seed) {
        document.getElementById('seed').value = currentImage.seed;
        showSuccess('Seed applied to form');
    }
}

async function showStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        const statsHtml = `
            <table class="table table-sm stats-table">
                <tr>
                    <td>Total Requests</td>
                    <td>${data.stats.total_requests}</td>
                </tr>
                <tr>
                    <td>Completed</td>
                    <td>${data.stats.total_completed}</td>
                </tr>
                <tr>
                    <td>Average Time</td>
                    <td>${data.stats.average_time.toFixed(2)}s</td>
                </tr>
                <tr>
                    <td>Queue Length</td>
                    <td>${data.queue_length}</td>
                </tr>
                <tr>
                    <td>Active GPUs</td>
                    <td>${data.num_gpus}</td>
                </tr>
            </table>
            <h6 class="mt-3">Popular Dimensions</h6>
            <ul class="list-unstyled">
                ${Object.entries(data.stats.dimension_counts || {})
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([dim, count]) => `<li>${dim}: ${count} times</li>`)
                    .join('')}
            </ul>
        `;
        
        document.getElementById('statsContent').innerHTML = statsHtml;
        
    } catch (error) {
        document.getElementById('statsContent').innerHTML = 
            '<p class="text-danger">Failed to load statistics</p>';
    }
    
    const modal = new bootstrap.Modal(document.getElementById('statsModal'));
    modal.show();
}

// Helper functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999;';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    toast.style.cssText = 'min-width: 250px; margin-bottom: 10px;';
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 150);
    }, 5000);
}

function showError(message) {
    showToast(message, 'error');
}

function showSuccess(message) {
    showToast(message, 'success');
}