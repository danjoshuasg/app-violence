# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from oauthlib.oauth2 import WebApplicationClient
import requests
from model.model import predict_violence  # Importa la función de predicción del modelo
import json
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

# Definición de extensiones permitidas y carpeta de subida
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Permitir transporte inseguro solo en desarrollo
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', 'TU_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', 'TU_CLIENT_SECRET')
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

client = WebApplicationClient(GOOGLE_CLIENT_ID)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/login')
def login():
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    redirect_uri = url_for('callback', _external=True)
    print("Redirect URI:", redirect_uri)  # Agregar para depuración

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route('/login/callback')
def callback():
    # Obtener el código de autorización de la URL
    code = request.args.get("code")

    # Descubrir el proveedor de OAuth 2.0 de Google
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Preparar y enviar una solicitud para obtener el token
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    # Parsear la respuesta de token
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Obtener información del usuario
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    # Almacenar la información del usuario en la sesión
    userinfo = userinfo_response.json()
    session['email'] = userinfo['email']
    session['name'] = userinfo['name']

    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/demo')
def demo():
    if 'email' in session:
        return redirect(url_for('home'))
    else:
        return render_template('demo.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'email' not in session:
        return jsonify({'error': 'Usuario no autenticado'}), 401

    video_file = request.files.get('video_file')
    if video_file and allowed_file(video_file.filename):
        # Asegurarse de que el directorio 'uploads' existe
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Obtener un nombre de archivo seguro
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Guardar el archivo
        video_file.save(video_path)

        # Reducir la resolución y convertir a MP4
        clip = VideoFileClip(video_path)
        clip_resized = clip.resize(height=720)  # Ajustar la altura a 720p
        converted_filename = os.path.splitext(filename)[0] + "_converted.mp4"
        converted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
        clip_resized.write_videofile(converted_video_path, codec="libx264")

        # Actualizar el nombre y ruta del video
        filename = converted_filename
        video_path = converted_video_path

        # Eliminar el archivo original si lo deseas
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename))

        # Devolver la URL del video convertido
        video_url = url_for('uploaded_file', filename=filename)

        return jsonify({
            'filename': filename,
            'video_url': video_url
        })
    except Exception as e:
        error = f"Ha ocurrido un error al cargar el video: {str(e)}"
        return jsonify({'error': error}), 500
    else:
        return jsonify({'error': 'Por favor, sube un archivo de video válido.'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'email' not in session:
        return jsonify({'error': 'Usuario no autenticado'}), 401

    data = request.get_json()
    filename = data.get('filename')

    if filename and allowed_file(filename):
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Realizar predicción utilizando la función importada
            _, prediction = predict_violence(video_path)

            return jsonify({
                'prediction': prediction
            })
        except Exception as e:
            error = f"Ha ocurrido un error al analizar el video: {str(e)}"
            return jsonify({'error': error}), 500
    else:
        return jsonify({'error': 'Nombre de archivo no válido.'}), 400

# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/home')
def home():
    if 'email' not in session:
        return redirect(url_for('index'))

    return render_template('home.html', user=session['name'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    #app.run(debug=True)
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=True)
