// script.js
document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const videoInput = document.getElementById('video_file');
    const uploadArea = document.getElementById('upload-area');
    const uploadLoading = document.getElementById('upload-loading');
    const videoContainer = document.getElementById('video-container');
    const uploadedVideo = document.getElementById('uploaded-video');
    const analyzeArea = document.getElementById('analyze-area');
    const analyzeButton = document.getElementById('analyze-button');
    const analysisLoading = document.getElementById('analysis-loading');
    const resultContainer = document.getElementById('result');
    const predictionText = document.getElementById('prediction');
    const filenameDisplay = document.getElementById('filename-display');


    let uploadedVideoFilename = '';

    // Debug: Verificar que todos los elementos existen
    console.log('DOM Elements check:', {
        uploadForm: !!uploadForm,
        videoInput: !!videoInput,
        uploadArea: !!uploadArea,
        videoContainer: !!videoContainer,
        uploadedVideo: !!uploadedVideo
    });

    // Añadir event listeners para debug del video
    uploadedVideo.addEventListener('loadstart', () => {
        console.log('Video loadstart event fired');
    });

    uploadedVideo.addEventListener('loadeddata', () => {
        console.log('Video loadeddata event fired');
        console.log('Video metadata:', {
            duration: uploadedVideo.duration,
            videoWidth: uploadedVideo.videoWidth,
            videoHeight: uploadedVideo.videoHeight,
            readyState: uploadedVideo.readyState
        });
    });

    uploadedVideo.addEventListener('error', (e) => {
        console.error('Video error event:', {
            error: uploadedVideo.error,
            networkState: uploadedVideo.networkState,
            errorCode: uploadedVideo.error ? uploadedVideo.error.code : null,
            errorMessage: uploadedVideo.error ? uploadedVideo.error.message : null
        });
    });

    // Manejar la selección de archivo
    videoInput.addEventListener('change', async function() {
        const file = this.files[0];
        
        if (file) {
            console.log('File selected:', {
                name: file.name,
                type: file.type,
                size: file.size
            });
            
            filenameDisplay.textContent = file.name;
            await handleUpload(file);
        }
    });

    // Función para manejar la carga del archivo
    async function handleUpload(file) {
        try {
            console.log('Starting upload process');
            uploadLoading.style.display = 'block';
            videoContainer.style.display = 'none';
            analyzeArea.style.display = 'none';
            resultContainer.style.display = 'none';

            const formData = new FormData();
            formData.append('video_file', file);

            console.log('Sending fetch request to /upload');
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            console.log('Server response status:', response.status);
            const data = await response.json();
            console.log('Server response data:', data);

            if (!response.ok) {
                throw new Error(data.error || 'Error al cargar el video');
            }

            // Actualizar UI con el video cargado
            uploadedVideoFilename = data.filename;
            console.log('Setting video source to:', data.video_url);
            uploadedVideo.src = data.video_url;
            
            // Asegurarse de que el video se cargue correctamente
            uploadedVideo.load();
            
            // Verificar el estado del video después de establecer el src
            console.log('Video element state after src set:', {
                src: uploadedVideo.src,
                currentSrc: uploadedVideo.currentSrc,
                readyState: uploadedVideo.readyState,
                networkState: uploadedVideo.networkState,
                paused: uploadedVideo.paused,
                error: uploadedVideo.error
            });

            videoContainer.style.display = 'block';
            analyzeArea.style.display = 'block';
            
        } catch (error) {
            console.error('Upload error:', error);
            alert(error.message);
        } finally {
            uploadLoading.style.display = 'none';
        }
    }

    // Funcionalidad de arrastrar y soltar
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadArea.classList.add('highlight');
    }

    function unhighlight() {
        uploadArea.classList.remove('highlight');
    }

    uploadArea.addEventListener('drop', async function(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];

        if (file && file.type.startsWith('video/')) {
            filenameDisplay.textContent = file.name;
            await handleUpload(file);
        } else {
            alert('Por favor, selecciona un archivo de video válido.');
        }
    }, false);

    // Manejar el análisis del video
    analyzeButton.addEventListener('click', async function () {
        try {
            analysisLoading.style.display = 'block';
            resultContainer.style.display = 'none';

            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filename: uploadedVideoFilename })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Error al analizar el video');
            }

            // Mostrar resultados
            predictionText.textContent = JSON.stringify(data.prediction, null, 2);
            resultContainer.style.display = 'block';

        } catch (error) {
            alert(error.message);
            console.error('Error:', error);
        } finally {
            analysisLoading.style.display = 'none';
        }
    });
});