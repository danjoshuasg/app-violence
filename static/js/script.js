document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const videoInput = document.getElementById('video_file');
    const uploadArea = document.getElementById('upload-area');
    const videoContainer = document.getElementById('video-container');
    const uploadedVideo = document.getElementById('uploaded-video');
    const contentTypeSelect = document.getElementById('content-type');
    const videoTypeSelect = document.getElementById('video-type');
    const filenameDisplay = document.getElementById('filename-display');
    const analysisLoading = document.getElementById('analysis-loading');
    const resultContainer = document.getElementById('result');
    const predictionText = document.getElementById('prediction');
    const analyzeArea = document.getElementById('analyze-area');
    const analyzeButton = document.getElementById('analyze-button');
    const modelTypeSelect = document.getElementById('model-type');

    // Mapa de opciones a rutas de video preexistentes
    const videoPaths = {
        'violencia': {
            'Espacios internos': 'violence_indoors.mp4',
            'Espacios externos': 'violence_outdoors.mp4',
            'En multitud': 'violence_crowded.mp4',
            'Nocturna': 'violence_night.mp4'
        },
        'no_violencia': {
        
            'Espacios abiertos': 'no_violence_outdoors.mp4',
            'En multitud': 'no_violence_crowded.mp4',
            'Afecto': 'no_violence_love.mp4'

        }
    };

    let customVideoLoaded = false; // Bandera para rastrear si se subió un video

    // Actualizar la lista de tipos de video
    function populateVideoTypes(contentType) {
        videoTypeSelect.innerHTML = '';
        const options = Object.keys(videoPaths[contentType]);
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            videoTypeSelect.appendChild(opt);
        });

        // Mostrar el video relacionado con la primera opción
        showVideo(contentType, options[0]);
    }

    // Mostrar video basado en tipo de contenido y tipo de video
    function showVideo(contentType, videoType) {
        if (!customVideoLoaded) {
            const videoFilename = videoPaths[contentType][videoType];
            uploadedVideo.src = `/videos/${videoFilename}`;
            uploadedVideo.load();
            videoContainer.hidden = false;
            analyzeArea.hidden = false; // Mostrar el área de análisis
            console.log('Analyze area hidden:', analyzeArea.hidden);
            console.log('Analyze area hidden after change:', analyzeArea.hidden);
        }
    }

    // Cambiar el video cuando se selecciona un nuevo tipo de video
    videoTypeSelect.addEventListener('change', function () {
        const contentType = contentTypeSelect.value;
        const videoType = videoTypeSelect.value;
        showVideo(contentType, videoType);
    });

    // Cambiar las opciones y el video al cambiar el tipo de contenido
    contentTypeSelect.addEventListener('change', function () {
        const contentType = contentTypeSelect.value;
        populateVideoTypes(contentType);
    });

    // Manejar la carga de un archivo de video personalizado
    videoInput.addEventListener('change', async function () {
        const file = this.files[0];
        if (file) {
            try {
                customVideoLoaded = true; // Marca que se cargó un video personalizado
                filenameDisplay.textContent = file.name;

                const formData = new FormData();
                formData.append('video_file', file);

                // Enviar el archivo al servidor
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Error al cargar el video');
                }

                // Actualizar la previsualización con el video subido
                uploadedVideo.src = data.video_url; // URL proporcionada por el servidor
                uploadedVideo.load();
                videoContainer.hidden = false;
                analyzeArea.hidden = false; // Mostrar el área de análisis
            } catch (error) {
                alert(error.message);
                console.error('Error al cargar el video:', error);
            }
        }
    });

    // Manejar el análisis del video
    analyzeButton.addEventListener('click', async function () {
        try {
            analysisLoading.hidden = false;
            resultContainer.hidden = true;

            let filename;
            if (customVideoLoaded) {
                // Cuando el video es subido por el usuario, 'uploadedVideo.src' es la URL
                // Extraemos solo el nombre del archivo
                const url = new URL(uploadedVideo.src);
                filename = url.pathname.split('/').pop();
            } else {
                // Cuando el video es preexistente, extraemos el nombre del archivo de la URL
                const url = new URL(uploadedVideo.src);
                filename = url.pathname.split('/').pop();
            }

            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    filename: filename, // Enviar solo el nombre del archivo
                    content_type: contentTypeSelect.value,
                    video_type: videoTypeSelect.value,
                    model_type: modelTypeSelect.value, // Agrega el tipo de modelo seleccionado
                    is_video_uploaded: customVideoLoaded
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Error al analizar el video');
            }

            // Mostrar resultados
            predictionText.textContent = JSON.stringify(data.prediction, null, 2);
            resultContainer.hidden = false;

        } catch (error) {
            alert(error.message);
            console.error('Error al analizar el video:', error);
        } finally {
            analysisLoading.hidden = true;
        }
    });

    // Inicializar opciones de video
    populateVideoTypes(contentTypeSelect.value);
});
