// Multi-Model Image Generation Studio - Frontend JavaScript

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
    "Enchanted forest path with glowing mushrooms and fireflies",
    "Portrait of a wise wizard with a crystal staff",
    "Futuristic space station orbiting Earth",
    "Medieval castle during a thunderstorm",
    "Tranquil beach at golden hour with palm trees",
    "Abstract colorful explosion of paint and light"
];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeWebSocket();
    loadHistory();
    setupEventListeners();
    updateModelDefaults();
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
        case 'model_status':
            // Update model status if needed
            break;
        case 'generation_start':
            // Could show which GPU/model is processing
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
    document.getElementById('width').addEventListener('input', validateDimensions);
    document.getElementById('height').addEventListener('input', validateDimensions);
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

// Model handling
function updateModelDefaults() {
    const modelSelect = document.getElementById('model');
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    
    if (!selectedOption) return;
    
    const modelKey = selectedOption.value;
    const modelInfo = MODELS_CONFIG[modelKey];
    
    if (!modelInfo) return;
    
    // Update description
    document.getElementById('modelDescription').textContent = modelInfo.description;
    
    // Update defaults
    document.getElementById('steps').value = selectedOption.dataset.steps;
    document.getElementById('stepsValue').textContent = selectedOption.dataset.steps;
    document.getElementById('guidance').value = selectedOption.dataset.guidance;
    document.getElementById('guidanceValue').textContent = selectedOption.dataset.guidance;
    
    // Update dimension limits
    const widthInput = document.getElementById('width');
    const heightInput = document.getElementById('height');
    
    widthInput.min = selectedOption.dataset.minWidth;
    widthInput.max = selectedOption.dataset.maxWidth;
    heightInput.min = selectedOption.dataset.minHeight;
    heightInput.max = selectedOption.dataset.maxHeight;
    
    // Update dimension presets based on model
    updateDimensionPresets(modelInfo);
    
    // Validate current dimensions
    validateDimensions();
    updateEstimatedTime();
}

function updateDimensionPresets(modelInfo) {
    const presetsContainer = document.getElementById('dimensionPresets');
    presetsContainer.innerHTML = '';
    
    // Common presets that fit within model limits
    const presets = [
        { width: 512, height: 512, label: '512²', icon: 'square' },
        { width: 768, height: 768, label: '768²', icon: 'square' },
        { width: 1024, height: 768, label: 'Landscape', icon: 'image' },
        { width: 768, height: 1024, label: 'Portrait', icon: 'phone' },
        { width: 1024, height: 1024, label: '1K²', icon: 'square-fill' },
        { width: 1536, height: 1024, label: 'Wide', icon: 'aspect-ratio' },
        { width: 1920, height: 1088, label: 'HD', icon: 'tv' },
    ];
    
    presets.forEach(preset => {
        if (preset.width >= modelInfo.min_width && preset.width <= modelInfo.max_width &&
            preset.height >= modelInfo.min_height && preset.height <= modelInfo.max_height) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-sm btn-outline-primary';
            btn.onclick = () => setDimensions(preset.width, preset.height);
            btn.innerHTML = `<i class="bi bi-${preset.icon}"></i> ${preset.label}`;
            presetsContainer.appendChild(btn);
        }
    });
}

function validateDimensions() {
    const modelSelect = document.getElementById('model');
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    
    if (!selectedOption) return;
    
    const width = parseInt(document.getElementById('width').value);
    const height = parseInt(document.getElementById('height').value);
    
    const minWidth = parseInt(selectedOption.dataset.minWidth);
    const maxWidth = parseInt(selectedOption.dataset.maxWidth);
    const minHeight = parseInt(selectedOption.dataset.minHeight);
    const maxHeight = parseInt(selectedOption.dataset.maxHeight);
    
    const warning = document.getElementById('dimensionWarning');
    
    if (width < minWidth || width > maxWidth || height < minHeight || height > maxHeight) {
        warning.textContent = `(Model supports ${minWidth}-${maxWidth} × ${minHeight}-${maxHeight})`;
    } else {
        warning.textContent = '';
    }
}

// Form handling
async function handleGenerate(e) {
    e.preventDefault();
    
    if (isGenerating) return;
    
    const formData = new FormData(e.target);
    const data = {
        prompt: formData.get('prompt'),
        model: formData.get('model'),
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
    
    startGeneration(data.model);
    
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

function startGeneration(modelKey) {
    isGenerating = true;
    generationStartTime = Date.now();
    
    // Update UI
    document.getElementById('generateBtn').classList.add('loading');
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('resultContainer').classList.add('d-none');
    document.getElementById('progressContainer').classList.remove('d-none');
    
    // Show which model is being used
    const modelName = MODELS_CONFIG[modelKey]?.name || modelKey;
    document.getElementById('progressModel').textContent = modelName;
    
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
                        <p class="mb-1"><strong>Model:</strong> ${result.model_name}</p>
                        <p class="mb-1"><strong>Prompt:</strong> ${escapeHtml(params.prompt)}</p>
                        <p class="mb-1"><strong>Dimensions:</strong> ${params.width}×${params.height}</p>
                    </div>
                    <div class="col-md-6">
                        <p class="mb-1"><strong>Steps:</strong> ${params.steps}</p>
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

// Model status display
async function showModels() {
    try {
        const response = await fetch('/api/models');
        const models = await response.json();
        
        let html = '<div class="table-responsive"><table class="table table-sm">';
        html += '<thead><tr><th>Model</th><th>Status</th><th>Workers</th><th>Queue</th><th>Dimensions</th></tr></thead><tbody>';
        
        for (const [key, info] of Object.entries(models)) {
            const statusBadge = info.loaded 
                ? `<span class="badge bg-success">Loaded</span>`
                : `<span class="badge bg-secondary">Not Loaded</span>`;
            
            const dimRange = `${info.min_width}-${info.max_width} × ${info.min_height}-${info.max_height}`;
            
            html += `
                <tr>
                    <td>
                        <strong>${info.name}</strong><br>
                        <small class="text-muted">${info.description}</small>
                    </td>
                    <td>${statusBadge}</td>
                    <td>${info.worker_count}</td>
                    <td>${info.queue_size}</td>
                    <td><small>${dimRange}</small></td>
                </tr>
            `;
        }
        
        html += '</tbody></table></div>';
        
        document.getElementById('modelsContent').innerHTML = html;
        
    } catch (error) {
        document.getElementById('modelsContent').innerHTML = 
            '<p class="text-danger">Failed to load model information</p>';
    }
    
    const modal = new bootstrap.Modal(document.getElementById('modelsModal'));
    modal.show();
}

// Stats display
async function showStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        let html = '<h6>Overall Statistics</h6>';
        html += `
            <table class="table table-sm stats-table mb-4">
                <tr>
                    <td>Total Requests</td>
                    <td>${data.total_requests}</td>
                </tr>
                <tr>
                    <td>Total Completed</td>
                    <td>${data.total_completed}</td>
                </tr>
            </table>
        `;
        
        if (data.models && Object.keys(data.models).length > 0) {
            html += '<h6>Model Statistics</h6>';
            
            for (const [modelKey, stats] of Object.entries(data.models)) {
                const modelName = MODELS_CONFIG[modelKey]?.name || modelKey;
                html += `
                    <div class="mb-3">
                        <h6 class="text-primary">${modelName}</h6>
                        <table class="table table-sm">
                            <tr>
                                <td>Requests</td>
                                <td>${stats.total_requests}</td>
                            </tr>
                            <tr>
                                <td>Average Time</td>
                                <td>${stats.average_time.toFixed(2)}s</td>
                            </tr>
                        </table>
                    </div>
                `;
            }
        }
        
        document.getElementById('statsContent').innerHTML = html;
        
    } catch (error) {
        document.getElementById('statsContent').innerHTML = 
            '<p class="text-danger">Failed to load statistics</p>';
    }
    
    const modal = new bootstrap.Modal(document.getElementById('statsModal'));
    modal.show();
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

async function addToHistory(result, params) {
    // Create a thumbnail to save space in localStorage
    const thumbnail = await createThumbnail(result.image, 128, 128);
    
    const historyItem = {
        id: Date.now(),
        thumbnail: thumbnail,
        params: params,
        seed: result.seed,
        gen_time: result.gen_time,
        model: result.model,
        model_name: result.model_name,
        timestamp: new Date().toISOString()
    };
    
    generationHistory.push(historyItem);
    saveHistory();
    displayHistory();
}

async function createThumbnail(base64Image, maxWidth, maxHeight) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = function() {
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
            resolve(canvas.toDataURL('image/jpeg', 0.8).split(',')[1]);
        };
        
        img.src = 'data:image/png;base64,' + base64Image;
    });
}

function displayHistory() {
    const container = document.getElementById('historyContainer');
    
    if (generationHistory.length === 0) {
        container.innerHTML = '<p class="text-muted text-center p-3">No history yet</p>';
        return;
    }
    
    const html = generationHistory.slice().reverse().map(item => `
        <div class="history-item" onclick="loadFromHistory('${item.id}')" 
             title="${escapeHtml(item.params.prompt)} (${item.model_name || item.model})">
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
    
    // Check if model is available
    if (item.model && LOADED_MODELS.includes(item.model)) {
        document.getElementById('model').value = item.model;
        updateModelDefaults();
    }
    
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
    validateDimensions();
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
    const model = document.getElementById('model').value;
    
    // Simple estimation formula (adjust based on your hardware)
    const pixels = width * height;
    let baseTime = 0.5; // Base time in seconds
    
    // Adjust base time by model
    if (model === 'flux-dev') baseTime = 1.0;
    else if (model === 'sdxl') baseTime = 0.8;
    
    const pixelFactor = pixels / 1000000; // Per million pixels
    const stepFactor = steps / 4; // Normalized to 4 steps
    
    const estimated = baseTime + (pixelFactor * 0.3) + (stepFactor * 0.2);
    
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
    const modelName = currentImage.model || 'unknown';
    link.download = `generated_${modelName}_${Date.now()}.png`;
    link.click();
}

function copySettings() {
    const settings = {
        model: document.getElementById('model').value,
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