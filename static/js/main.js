document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    const previewWrapper = document.getElementById('preview-wrapper');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImg = document.getElementById('btn-remove-img');
    const btnPredict = document.getElementById('btn-predict');
    const btnTrain = document.getElementById('btn-train-pipeline');
    
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultContent = document.getElementById('result-content');
    
    const diagName = document.getElementById('diag-name');
    const riskBadge = document.getElementById('risk-badge');
    const confidenceVal = document.getElementById('confidence-val');
    const confidenceBar = document.getElementById('confidence-bar');
    const diagDescription = document.getElementById('diag-description');
    const probList = document.getElementById('prob-list');
    
    const toast = document.getElementById('status-toast');
    const toastMsg = document.getElementById('toast-message');

    let selectedFile = null;
    let selectedSamplePath = null;

    // Trigger File Picker
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('#btn-remove-img') || selectedFile || selectedSamplePath) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('drag-over');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFileSelect(dt.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        selectedSamplePath = null;
        clearSampleSelection();

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Remove Image
    btnRemoveImg.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        selectedSamplePath = null;
        fileInput.value = '';
        imagePreview.src = '#';
        previewWrapper.classList.add('hidden');
        dropzonePrompt.classList.remove('hidden');
        btnPredict.disabled = true;
        clearSampleSelection();
    }

    // Sample Selection
    const sampleItems = document.querySelectorAll('.sample-item');
    sampleItems.forEach(item => {
        item.addEventListener('click', (e) => {
            clearSampleSelection();
            item.classList.add('active');
            selectedSamplePath = item.getAttribute('data-path');
            selectedFile = null;
            
            imagePreview.src = "/" + selectedSamplePath;
            dropzonePrompt.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
            btnPredict.disabled = false;
        });
    });

    function clearSampleSelection() {
        sampleItems.forEach(item => item.classList.remove('active'));
    }

    // Predict Action
    btnPredict.addEventListener('click', async () => {
        if (!selectedFile && !selectedSamplePath) return;

        showLoading(true);

        try {
            let response;
            if (selectedFile) {
                const formData = new FormData();
                formData.append('image', selectedFile);
                response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
            } else {
                response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_path: selectedSamplePath })
                });
            }

            const data = await response.json();
            showLoading(false);

            if (data && data[0]) {
                displayResult(data[0]);
            } else {
                showToast('Failed to analyze image.', 'error');
            }
        } catch (err) {
            console.error(err);
            showLoading(false);
            showToast('Error connecting to prediction server.', 'error');
        }
    });

    function displayResult(result) {
        emptyState.classList.add('hidden');
        resultContent.classList.remove('hidden');

        diagName.textContent = result.display_name;
        confidenceVal.textContent = `${result.confidence}%`;
        confidenceBar.style.width = `${result.confidence}%`;
        diagDescription.textContent = result.description;

        if (result.prediction === 'normal') {
            riskBadge.textContent = 'Low Risk';
            riskBadge.className = 'badge badge-risk-normal';
        } else {
            riskBadge.textContent = 'High Risk';
            riskBadge.className = 'badge badge-risk-high';
        }

        // Render breakdown bars
        probList.innerHTML = '';
        result.probabilities.forEach(item => {
            const isTop = item.class_key === result.prediction;
            const itemHtml = `
                <div class="prob-item">
                    <div class="prob-info">
                        <span>${item.label}</span>
                        <strong>${item.probability}%</strong>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: ${item.probability}%; background: ${isTop ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.2)'};"></div>
                    </div>
                </div>
            `;
            probList.insertAdjacentHTML('beforeend', itemHtml);
        });
    }

    function showLoading(isLoading) {
        if (isLoading) {
            emptyState.classList.add('hidden');
            resultContent.classList.add('hidden');
            loadingState.classList.remove('hidden');
        } else {
            loadingState.classList.add('hidden');
        }
    }

    // Trigger Training
    btnTrain.addEventListener('click', async () => {
        try {
            showToast('Triggering ML training pipeline...', 'info');
            const res = await fetch('/train', { method: 'POST' });
            const data = await res.json();
            showToast(data.message, 'success');
        } catch (e) {
            showToast('Failed to trigger training pipeline.', 'error');
        }
    });

    function showToast(msg, type = 'info') {
        toastMsg.textContent = msg;
        toast.classList.remove('hidden');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 5000);
    }
});
