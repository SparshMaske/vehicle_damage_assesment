document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultsSection = document.getElementById('results');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('results-content');
    const uploadContent = document.querySelector('.upload-content');

    let currentFile = null;

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // Click handler
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.classList.remove('hidden');
                    uploadContent.classList.add('hidden');
                    analyzeBtn.disabled = false;
                };
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file (JPEG, PNG, WEBP).');
            }
        }
    }

    // Analyze button handler
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI state changes
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('hidden');
        resetBtn.classList.remove('hidden');
        dropZone.style.pointerEvents = 'none';
        
        resultsSection.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        loader.classList.remove('hidden');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            // Note: In development, this points to our proxy server running on port 3000
            // which forwards to the FastAPI backend on port 8000
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to analyze image');
            }

            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
            resetApp();
        }
    });

    // Reset button handler
    resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        
        analyzeBtn.disabled = true;
        analyzeBtn.classList.remove('hidden');
        resetBtn.classList.add('hidden');
        dropZone.style.pointerEvents = 'auto';
        
        resultsSection.classList.add('hidden');
    }

    function displayResults(data) {
        // Hide loader, show results
        loader.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Set annotated image
        document.getElementById('annotated-image').src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
        
        // Update mode badge
        document.getElementById('mode-badge').textContent = data.processing_mode || 'Live Inference';

        // Update routing decision
        const routingEl = document.getElementById('res-routing');
        routingEl.textContent = data.routing_decision;
        routingEl.className = 'card-value ' + getRoutingClass(data.routing_decision);

        // Update overall severity
        const severityEl = document.getElementById('res-severity');
        severityEl.textContent = data.overall_severity;
        severityEl.className = 'card-value ' + getSeverityClass(data.overall_severity);

        // Update cost
        document.getElementById('res-cost').textContent = data.estimated_cost_range;

        // Update reasoning
        document.getElementById('res-reasoning').textContent = data.reasoning;

        // Update detections list
        const detectionsList = document.getElementById('res-detections');
        document.getElementById('res-count').textContent = data.damage_detections.length;
        
        detectionsList.innerHTML = '';
        
        if (data.damage_detections.length === 0) {
            detectionsList.innerHTML = '<li class="detection-item" style="justify-content:center; color: var(--text-secondary)">No damage detected</li>';
        } else {
            data.damage_detections.forEach(det => {
                const li = document.createElement('li');
                li.className = 'detection-item';
                
                const confidence = (det.confidence * 100).toFixed(1);
                
                li.innerHTML = `
                    <div class="detection-info">
                        <span class="detection-type">${det.type.replace('_', ' ')}</span>
                        <span class="detection-conf">Confidence: ${confidence}%</span>
                    </div>
                    <span class="detection-severity" style="color: ${getSeverityColor(det.severity)}">
                        ${det.severity}
                    </span>
                `;
                detectionsList.appendChild(li);
            });
        }
    }

    // Helpers
    function getRoutingClass(routing) {
        if (routing.includes('Straight-Through')) return 'success';
        if (routing.includes('Needs Adjuster')) return 'danger';
        return 'warning';
    }

    function getSeverityClass(severity) {
        switch(severity.toLowerCase()) {
            case 'minor': return 'success';
            case 'moderate': return 'warning';
            case 'severe': return 'danger';
            default: return '';
        }
    }

    function getSeverityColor(severity) {
        switch(severity.toLowerCase()) {
            case 'minor': return 'var(--success)';
            case 'moderate': return 'var(--warning)';
            case 'severe': return 'var(--danger)';
            default: return 'var(--text-primary)';
        }
    }
});
