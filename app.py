import os
import logging
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from oauthlib.oauth2 import WebApplicationClient
import requests
from model.model import predict_violence  # Importa la función de predicción del modelo
import json
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from urllib.parse import urlparse
import time

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,  # Cambia a DEBUG para más detalles
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Registros en un archivo
        logging.StreamHandler()          # Registros en la consola
    ]
)

load_dotenv()
logging.info("Configuración de variables de entorno cargada.")

# Definición de extensiones permitidas y carpetas
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
VIDEOS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEOS_FOLDER'] = VIDEOS_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['PREFERRED_URL_SCHEME'] = 'https'
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Permitir transporte inseguro solo en desarrollo
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
logging.debug("OAUTHLIB_INSECURE_TRANSPORT configurado a '1'.")

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', 'TU_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', 'TU_CLIENT_SECRET')

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

client = WebApplicationClient(GOOGLE_CLIENT_ID)

def allowed_file(filename):
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    logging.debug(f"Verificando si el archivo '{filename}' está permitido: {is_allowed}")
    return is_allowed

@app.route('/')
def index():
    logging.info("Accediendo a la ruta '/'")
    if 'email' in session:
        logging.debug("Usuario autenticado encontrado en la sesión, redirigiendo a 'home'.")
        return redirect(url_for('home'))
    logging.debug("Usuario no autenticado, renderizando 'index.html'.")
    return render_template('index.html')

@app.route('/login')
def login():
    logging.info("Accediendo a la ruta '/login'")
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Agregar para depuración
    redirect_uri = url_for('callback', _external=True)
    logging.debug(f"Redirect URI generada: {redirect_uri}")
    logging.debug(f"Host URL: {request.host_url}")
    logging.debug(f"URL Root: {request.url_root}")
    logging.debug(f"Esquema de la solicitud: {request.scheme}")

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope=["openid", "email", "profile"],
    )
    logging.debug(f"URI de solicitud preparada para OAuth: {request_uri}")
    return redirect(request_uri)

@app.route('/login/callback')
def callback():
    logging.info("Accediendo a la ruta '/login/callback'")
    # Obtener el código de autorización de la URL
    code = request.args.get("code")
    logging.debug(f"Código de autorización recibido: {code}")

    # Descubrir el proveedor de OAuth 2.0 de Google
    google_provider_cfg = requests.get(GOOGLE_DISCOVERY_URL).json()
    token_endpoint = google_provider_cfg["token_endpoint"]
    logging.debug(f"Token endpoint descubierto: {token_endpoint}")

    # Preparar y enviar una solicitud para obtener el token
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )
    logging.debug("Solicitud de token preparada, enviando solicitud POST.")
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )
    logging.debug(f"Respuesta de token recibida: {token_response.status_code}")

    # Parsear la respuesta de token
    client.parse_request_body_response(json.dumps(token_response.json()))
    logging.debug("Respuesta de token parseada exitosamente.")

    # Obtener información del usuario
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    logging.debug("Solicitando información del usuario con token.")
    userinfo_response = requests.get(uri, headers=headers, data=body)
    logging.debug(f"Respuesta de información del usuario: {userinfo_response.status_code}")

    # Almacenar la información del usuario en la sesión
    userinfo = userinfo_response.json()
    session['email'] = userinfo.get('email')
    session['name'] = userinfo.get('name')
    logging.info(f"Usuario autenticado: {session.get('email')}")

    return redirect(url_for('home'))

@app.route('/about')
def about():
    logging.info("Accediendo a la ruta '/about'")
    return render_template('about.html')

@app.route('/demo')
def demo():
    logging.info("Accediendo a la ruta '/demo'")
    if 'email' in session:
        logging.debug("Usuario autenticado encontrado en la sesión, redirigiendo a 'home'.")
        return redirect(url_for('home'))
    logging.debug("Usuario no autenticado, renderizando 'demo.html'.")
    return render_template('demo.html')

@app.route('/videos/<filename>')
def videos(filename):
    logging.info(f"Accediendo a la ruta '/videos/{filename}'")
    try:
        # Asegurar que existe el directorio de uploads
        if not os.path.exists(app.config['VIDEOS_FOLDER']):
            os.makedirs(app.config['VIDEOS_FOLDER'])
            logging.debug(f"Directorio de uploads creado: {app.config['VIDEOS_FOLDER']}")
        return send_from_directory(app.config['VIDEOS_FOLDER'], filename)
    except Exception as e:
        logging.error(f"Error al servir el video '{filename}': {e}")
        return jsonify({'error': 'Archivo de video no encontrado.'}), 404

@app.route('/upload', methods=['POST'])
def upload():
    logging.info("Recibiendo solicitud de carga de video en '/upload'")
    if 'email' not in session:
        logging.warning("Intento de carga sin autenticación.")
        return jsonify({'error': 'Usuario no autenticado'}), 401

    video_file = request.files.get('video_file')
    if not video_file or not allowed_file(video_file.filename):
        logging.warning("Archivo de video no válido o ausente en la solicitud de carga.")
        return jsonify({'error': 'Por favor, sube un archivo de video válido.'}), 400

    try:
        # Asegurar que existe el directorio de uploads
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            logging.debug(f"Directorio de uploads creado: {app.config['UPLOAD_FOLDER']}")

        # Preparar nombres de archivo
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        converted_filename = os.path.splitext(filename)[0] + "_converted.mp4"
        converted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)

        logging.debug(f"Guardando archivo original en: {video_path}")
        # Guardar archivo original
        video_file.save(video_path)
        logging.info(f"Archivo de video guardado: {video_path}")

        # Convertir video con un timeout de 60 segundos
        max_wait = 60  # segundos
        start_time = time.time()
        logging.debug("Iniciando conversión de video.")

        try:
            clip = VideoFileClip(video_path)
            clip_resized = clip.resize(height=720)
            clip_resized.write_videofile(converted_video_path, codec="libx264")
            clip.close()
            clip_resized.close()
            logging.info(f"Video convertido exitosamente: {converted_video_path}")
        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(f"Error durante la conversión del video: {e}")
            if elapsed_time >= max_wait:
                logging.warning("Timeout alcanzado durante la conversión, usando video original.")
                return jsonify({
                    'filename': filename,
                    'video_url': url_for('uploaded_file', filename=filename),
                    'warning': 'Se está usando el video original debido a timeout en la conversión.'
                })
            raise e

        # Eliminar el archivo original
        try:
            os.remove(video_path)
            logging.debug(f"Archivo original eliminado: {video_path}")
        except Exception as e:
            logging.error(f"Error al eliminar el archivo original '{video_path}': {e}")

        return jsonify({
            'filename': converted_filename,
            'video_url': url_for('uploaded_file', filename=converted_filename)
        })

    except Exception as e:
        error_message = f"Error en la carga del video: {str(e)}"
        logging.error(error_message)
        # Limpiar archivos en caso de error
        for path in [video_path, converted_video_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logging.debug(f"Archivo eliminado durante limpieza: {path}")
            except Exception as cleanup_error:
                logging.error(f"Error al limpiar el archivo '{path}': {cleanup_error}")
        return jsonify({'error': error_message}), 500

def extract_local_path(filename, base_folder):
    video_name = secure_filename(filename)
    local_path = os.path.join(base_folder, video_name)
    logging.debug(f"Ruta local extraída para '{filename}': {local_path}")
    return local_path

@app.route('/analyze', methods=['POST'])
def analyze():
    logging.info("Recibiendo solicitud de análisis de video en '/analyze'")
    if 'email' not in session:
        logging.warning("Intento de análisis sin autenticación.")
        return jsonify({'error': 'Usuario no autenticado'}), 401
    
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
        logging.debug(f"Directorio processed creado: {app.config['PROCESSED_FOLDER']}")
    
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type')
    is_video_uploaded = data.get('is_video_uploaded')
    logging.debug(f"Datos recibidos para análisis: filename={filename}, model_type={model_type}, is_video_uploaded={is_video_uploaded}")

    if filename and allowed_file(filename):
        if is_video_uploaded:
            base_folder = app.config['UPLOAD_FOLDER']
        else:
            base_folder = app.config['VIDEOS_FOLDER']

        video_path = extract_local_path(filename, base_folder)
        logging.debug(f"Ruta completa del video para análisis: {video_path}")

        if not os.path.exists(video_path):
            logging.error(f"El archivo de video no existe: {video_path}")
            return jsonify({'error': f'El archivo {filename} no existe en el servidor.'}), 404

        try:
            # Realizar predicción y obtener el video procesado
            processed_video_path, prediction = predict_violence(video_path, model_type)
            
            if processed_video_path:
                # Convertir el video procesado
                try:
                    logging.debug("Iniciando conversión del video procesado")
                    clip = VideoFileClip(processed_video_path)
                    clip_resized = clip.resize(height=720)
                    
                    # Crear nombre para el video convertido
                    processed_filename = os.path.basename(processed_video_path)
                    converted_filename = os.path.splitext(processed_filename)[0] + "_converted.mp4"
                    final_processed_path = os.path.join(app.config['PROCESSED_FOLDER'], converted_filename)
                    
                    # Convertir y guardar
                    clip_resized.write_videofile(final_processed_path, codec="libx264")
                    clip.close()
                    clip_resized.close()
                    
                    # Eliminar el video procesado original
                    if os.path.exists(processed_video_path):
                        os.remove(processed_video_path)
                        logging.debug(f"Video procesado original eliminado: {processed_video_path}")
                    
                    logging.info(f"Video procesado convertido y guardado en: {final_processed_path}")
                    
                    # Generar URL para el video procesado convertido
                    video_url = url_for('processed_file', filename=converted_filename)
                    
                    return jsonify({
                        'prediction': prediction,
                        'video_url': video_url
                    })
                    
                except Exception as conv_error:
                    logging.error(f"Error en la conversión del video procesado: {conv_error}")
                    # En caso de error en la conversión, intentamos usar el video original procesado
                    if os.path.exists(processed_video_path):
                        processed_filename = os.path.basename(processed_video_path)
                        final_processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                        os.rename(processed_video_path, final_processed_path)
                        video_url = url_for('processed_file', filename=processed_filename)
                        
                        return jsonify({
                            'prediction': prediction,
                            'video_url': video_url,
                            'warning': 'Se está usando el video sin convertir debido a un error en la conversión.'
                        })
                    else:
                        return jsonify({
                            'error': 'Error en la conversión y el video original no está disponible',
                            'prediction': prediction
                        }), 500
            else:
                return jsonify({
                    'error': 'No se pudo generar el video procesado',
                    'prediction': prediction
                }), 500
                
        except Exception as e:
            error = f"Ha ocurrido un error al analizar el video: {str(e)}"
            logging.error(error)
            return jsonify({'error': error}), 500
    else:
        logging.warning("Nombre de archivo no válido recibido para análisis.")
        return jsonify({'error': 'Nombre de archivo no válido.'}), 400

# Ruta para servir los archivos subidos
@app.route('/processed/<filename>')
def processed_file(filename):
    """Ruta para servir los archivos de video procesados."""
    logging.info(f"Accediendo a la ruta '/processed/{filename}'")
    try:
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    except Exception as e:
        logging.error(f"Error al servir el archivo procesado '{filename}': {e}")
        return jsonify({'error': 'Archivo procesado no encontrado.'}), 404
    
# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logging.info(f"Accediendo a la ruta '/uploads/{filename}'")
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logging.error(f"Error al servir el archivo subido '{filename}': {e}")
        return jsonify({'error': 'Archivo no encontrado.'}), 404

@app.route('/home')
def home():
    logging.info("Accediendo a la ruta '/home'")
    if 'email' not in session:
        logging.warning("Usuario no autenticado intentando acceder a 'home'.")
        return redirect(url_for('index'))

    logging.debug(f"Renderizando 'home.html' para el usuario: {session.get('name')}")
    return render_template('home.html', user=session['name'])

@app.route('/logout')
def logout():
    logging.info("Usuario cerrando sesión.")
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    logging.info("Iniciando la aplicación Flask en modo debug.")
    app.run(debug=True)
    #port = int(os.environ.get('PORT', 8050))
    #app.run_server(host='0.0.0.0', port=port, debug=True)
