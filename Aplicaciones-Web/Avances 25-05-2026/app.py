from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
from datetime import datetime
import threading
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import statistics
 
# ── VERIFICACIÓN DE CÉDULA ──────────────────────────────────
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
 
def verificar_cedula_en_foto(foto, cedula):
    """Lee el texto de la foto de cédula y verifica que el número esté presente."""
    try:
        imagen = Image.open(foto)
        texto = pytesseract.image_to_string(imagen)
        return cedula in texto
    except Exception as e:
        print(f"Error al verificar cédula: {e}")
        return False
# ────────────────────────────────────────────────────────────
 
def classify_fast(v):
    if v and v >= 126:
        return "Diabetes"
    elif v and v >= 100:
        return "Prediabetes"
    else:
        return "Normal"
 
def classify_post(v):
    if v and v >= 200:
        return "Diabetes"
    elif v and v >= 140:
        return "Prediabetes"
    else:
        return "Normal"
 
def classify_a1c(v):
    if v and v >= 6.5:
        return "Diabetes"
    elif v and v >= 5.7:
        return "Prediabetes"
    else:
        return "Normal"
 
# Importar módulos existentes
import sys
sys.path.append('..')
from diabetes_model import (
    assess_diabetes_risk_comprehensive,
    map_probability_to_timeframe,
    possible_symptoms_by_probability,
    classify_lipid_profile,
    generate_lipid_recommendations,
    AdvancedDiabetesModel,
)
 
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'  # Cambiar en producción
import mimetypes
mimetypes.add_type('text/css', '.css')
 
# Base de datos
DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')
 
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
def init_db():
    db = get_db()
    with open('schema.sql', 'r') as f:
        db.executescript(f.read())
    db.commit()
 
# Modelo de IA
diabetes_model = None
model_status = "No cargado"
 
def load_model():
    global diabetes_model, model_status
    try:
        model_path = "diabetes_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                diabetes_model = pickle.load(f)
            model_status = "Modelo listo"
        else:
            from diabetes_model import train_advanced_model
            diabetes_model = train_advanced_model()
            model_status = "Modelo entrenado"
    except Exception as e:
        model_status = f"Error: {str(e)}"
 
# Cargar modelo en background
threading.Thread(target=load_model, daemon=True).start()
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/register_patient', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        nombre = request.form.get('nombre', '').strip()
        fecha_nacimiento = request.form.get('fecha_nacimiento', '').strip()
        lugar_tratamiento = request.form.get('lugar_tratamiento', '').strip()
        nacionalidad = request.form.get('nacionalidad', '').strip()
        edad = request.form.get('edad', '').strip()
        email = request.form.get('email', '').strip()
        password_raw = request.form.get('password', '')
 
        if not (nombre and fecha_nacimiento and lugar_tratamiento and nacionalidad and edad and email and password_raw):
            flash('Por favor completa todos los campos requeridos.')
            return render_template('register_patient.html')
 
        try:
            edad = int(edad)
        except ValueError:
            flash('La edad debe ser un número válido.')
            return render_template('register_patient.html')
 
        password = generate_password_hash(password_raw)
        fecha_ingreso = datetime.now().strftime('%Y-%m-%d')
 
        db = get_db()
        try:
            db.execute('INSERT INTO pacientes (nombre, fecha_nacimiento, lugar_tratamiento, nacionalidad, edad, email, password, fecha_ingreso) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                       (nombre, fecha_nacimiento, lugar_tratamiento, nacionalidad, edad, email, password, fecha_ingreso))
            db.commit()
        except sqlite3.IntegrityError:
            flash('Este correo ya está registrado. Usa otro correo.')
            return render_template('register_patient.html')
 
        flash('Registro exitoso')
        return redirect(url_for('login'))
    return render_template('register_patient.html')
 
@app.route('/register_professional', methods=['GET', 'POST'])
def register_professional():
    if request.method == 'POST':
        cedula = request.form.get('cedula', '').strip()
        nombre = request.form.get('nombre', '').strip()
        edad = request.form.get('edad', '').strip()
        fecha_nacimiento = request.form.get('fecha_nacimiento', '').strip()
        lugar_trabajo = request.form.get('lugar_trabajo', '').strip()
        nacionalidad = request.form.get('nacionalidad', '').strip()
        curp = request.form.get('curp', '').strip()
        rfc = request.form.get('rfc', '').strip()
        clues = request.form.get('clues', '').strip()
        cct = request.form.get('cct', '').strip()
        password_raw = request.form.get('password', '')
 
        if not (cedula and nombre and edad and fecha_nacimiento and lugar_trabajo and nacionalidad and curp and rfc and password_raw):
            flash('Por favor completa todos los campos requeridos.')
            return render_template('register_professional.html')
 
        try:
            edad = int(edad)
        except ValueError:
            flash('La edad debe ser un número válido.')
            return render_template('register_professional.html')
 
        # ── VERIFICACIÓN DE CÉDULA ──
        foto_cedula = request.files.get('foto_cedula')
        if not foto_cedula:
            flash('Por favor sube una foto de tu cédula profesional.')
            return render_template('register_professional.html')
 
        if not verificar_cedula_en_foto(foto_cedula, cedula):
            flash('No pudimos verificar tu cédula en la foto. Asegúrate de que el número sea visible y legible.')
            return render_template('register_professional.html')
        # ────────────────────────────
 
        password = generate_password_hash(password_raw)
        fecha_ingreso = datetime.now().strftime('%Y-%m-%d')
 
        # Manejar foto de perfil
        foto = request.files.get('foto')
        foto_path = None
        if foto:
            foto_path = os.path.join('static/uploads', foto.filename)
            foto.save(foto_path)
 
        db = get_db()
        try:
            db.execute('INSERT INTO profesionales (cedula, nombre, edad, fecha_nacimiento, lugar_trabajo, nacionalidad, curp, rfc, clues, cct, password, foto_path, fecha_ingreso) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                       (cedula, nombre, edad, fecha_nacimiento, lugar_trabajo, nacionalidad, curp, rfc, clues, cct, password, foto_path, fecha_ingreso))
            db.commit()
        except sqlite3.IntegrityError:
            flash('Esta cédula ya está registrada.')
            return render_template('register_professional.html')
 
        flash('Registro exitoso')
        return redirect(url_for('login'))
    return render_template('register_professional.html')
 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form.get('identifier') or request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
 
        if not identifier or not password or not user_type:
            flash('Completa todos los campos para iniciar sesión.')
            return render_template('login.html')
 
        db = get_db()
        if user_type == 'paciente':
            user = db.execute('SELECT * FROM pacientes WHERE email = ?', (identifier,)).fetchone()
        else:
            user = db.execute('SELECT * FROM profesionales WHERE cedula = ? OR nombre = ?', (identifier, identifier)).fetchone()
 
        if user:
            print(f"Login intento: {user_type=} {identifier=} -> encontrado usuario id={user['id']}")
            if check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['user_type'] = user_type
                if user_type == 'paciente':
                    return redirect(url_for('patient_dashboard'))
                else:
                    return redirect(url_for('professional_dashboard'))
            else:
                flash('Contraseña incorrecta. Verifica tu contraseña y vuelve a intentarlo.')
        else:
            print(f"Login intento: {user_type=} {identifier=} -> NO encontrado")
            flash('Usuario no encontrado. Verifica que seleccionaste el tipo de usuario correcto y que escribiste bien tu correo/cédula/nombre.')
 
    return render_template('login.html')
 
@app.route('/patient_dashboard')
def patient_dashboard():
    if 'user_id' not in session or session['user_type'] != 'paciente':
        return redirect(url_for('login'))
    db = get_db()
    lecturas = db.execute('SELECT * FROM lecturas WHERE paciente_id = ? ORDER BY fecha DESC', (session['user_id'],)).fetchall()
    return render_template('patient_dashboard.html', lecturas=lecturas)
 
@app.route('/add_reading', methods=['POST'])
def add_reading():
    if 'user_id' not in session or session['user_type'] != 'paciente':
        return redirect(url_for('login'))
 
    dia = request.form.get('dia')
    glucosa_ayunas = request.form.get('glucosa_ayunas')
    glucosa_2h = request.form.get('glucosa_2h')
    a1c = request.form.get('a1c')
    ldl = request.form.get('ldl')
    hdl = request.form.get('hdl')
    trigliceridos = request.form.get('trigliceridos')
    colesterol_total = request.form.get('colesterol_total')
 
    try:
        dia = int(dia)
    except (ValueError, TypeError):
        flash('El día debe ser un número entero válido.')
        return redirect(url_for('patient_dashboard'))
 
    fecha = datetime.now().strftime('%Y-%m-%d')
 
    db = get_db()
    db.execute(
        'INSERT INTO lecturas (paciente_id, dia, fecha, glucosa_ayunas, glucosa_2h, a1c, ldl, hdl, trigliceridos, colesterol_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (session['user_id'], dia, fecha, glucosa_ayunas or None, glucosa_2h or None, a1c or None, ldl or None, hdl or None, trigliceridos or None, colesterol_total or None)
    )
    db.commit()
    flash('Lectura agregada correctamente.')
    return redirect(url_for('patient_dashboard'))
 
@app.route('/professional_dashboard', methods=['GET', 'POST'])
def professional_dashboard():
    if 'user_id' not in session or session['user_type'] != 'profesional':
        return redirect(url_for('login'))
 
    db = get_db()
    profesional = db.execute('SELECT * FROM profesionales WHERE id = ?', (session['user_id'],)).fetchone()
 
    if request.method == 'POST' and 'select_patient' in request.form:
        paciente_id = request.form.get('paciente_id')
        if paciente_id:
            session['current_patient_id'] = int(paciente_id)
        return redirect(url_for('professional_dashboard'))
 
    pacientes = db.execute('SELECT id, nombre FROM pacientes ORDER BY nombre').fetchall()
 
    current_patient_id = session.get('current_patient_id')
    paciente = None
    lecturas = []
    resumen = "Seleccione un paciente para ver resumen."
    info_paciente = "Seleccione un paciente para ver información."
 
    if current_patient_id:
        paciente = db.execute('SELECT * FROM pacientes WHERE id = ?', (current_patient_id,)).fetchone()
        if paciente:
            lecturas = [dict(r) for r in db.execute('SELECT * FROM lecturas WHERE paciente_id = ? ORDER BY dia', (current_patient_id,)).fetchall()]
            resumen = calcular_resumen_paciente(lecturas)
            info_paciente = f"Nombre: {paciente['nombre']}\nEzdad: {paciente['edad']}\nFecha de Nacimiento: {paciente['fecha_nacimiento']}\nLugar de Tratamiento: {paciente['lugar_tratamiento']}\nNacionalidad: {paciente['nacionalidad']}\nFecha de Ingreso: {paciente['fecha_ingreso']}"
 
    return render_template('professional_dashboard.html', pacientes=pacientes, paciente=paciente, lecturas=lecturas, resumen=resumen, info_paciente=info_paciente, model_status=model_status, classify_fast=classify_fast, classify_post=classify_post, classify_a1c=classify_a1c, profesional=profesional)
 
def calcular_resumen_paciente(lecturas):
    if not lecturas:
        return "No hay lecturas disponibles."
 
    fast_vals = [r['glucosa_ayunas'] for r in lecturas if r['glucosa_ayunas'] and r['glucosa_ayunas'] > 0]
    post_vals = [r['glucosa_2h'] for r in lecturas if r['glucosa_2h'] and r['glucosa_2h'] > 0]
    a1c_vals = [r['a1c'] for r in lecturas if r['a1c']]
    ldl_vals = [r['ldl'] for r in lecturas if r['ldl']]
    hdl_vals = [r['hdl'] for r in lecturas if r['hdl']]
    trig_vals = [r['trigliceridos'] for r in lecturas if r['trigliceridos']]
 
    resumen = "Resumen del Paciente:\n"
    if fast_vals:
        avg_fast = statistics.mean(fast_vals)
        resumen += f"Glucosa Ayunas Promedio: {avg_fast:.1f} mg/dL ({'Alto' if avg_fast >= 100 else 'Normal'})\n"
    if post_vals:
        avg_post = statistics.mean(post_vals)
        resumen += f"Glucosa 2h Post Promedio: {avg_post:.1f} mg/dL ({'Alto' if avg_post >= 140 else 'Normal'})\n"
    if a1c_vals:
        avg_a1c = statistics.mean(a1c_vals)
        resumen += f"A1C Promedio: {avg_a1c:.1f}% ({'Alto' if avg_a1c >= 5.7 else 'Normal'})\n"
    if ldl_vals:
        avg_ldl = statistics.mean(ldl_vals)
        resumen += f"LDL Promedio: {avg_ldl:.1f} mg/dL ({'Alto' if avg_ldl >= 160 else 'Normal'})\n"
    if hdl_vals:
        avg_hdl = statistics.mean(hdl_vals)
        resumen += f"HDL Promedio: {avg_hdl:.1f} mg/dL ({'Bajo' if avg_hdl < 40 else 'Normal'})\n"
    if trig_vals:
        avg_trig = statistics.mean(trig_vals)
        resumen += f"Triglicéridos Promedio: {avg_trig:.1f} mg/dL ({'Alto' if avg_trig >= 200 else 'Normal'})\n"
 
    return resumen
 
@app.route('/save_reading', methods=['POST'])
def save_reading():
    if 'user_id' not in session or session['user_type'] != 'profesional':
        return redirect(url_for('login'))
 
    current_patient_id = session.get('current_patient_id')
    if not current_patient_id:
        flash('Selecciona un paciente primero.')
        return redirect(url_for('professional_dashboard'))
 
    dia = request.form.get('dia')
    glucosa_ayunas = request.form.get('glucosa_ayunas')
    glucosa_2h = request.form.get('glucosa_2h')
    a1c = request.form.get('a1c')
    ldl = request.form.get('ldl')
    hdl = request.form.get('hdl')
    trigliceridos = request.form.get('trigliceridos')
    colesterol_total = request.form.get('colesterol_total')
 
    try:
        dia = int(dia)
        if not (1 <= dia <= 90):
            raise ValueError
    except:
        flash('Día inválido.')
        return redirect(url_for('professional_dashboard'))
 
    fecha = datetime.now().strftime('%Y-%m-%d')
 
    db = get_db()
    db.execute('''
        INSERT OR REPLACE INTO lecturas 
        (paciente_id, dia, fecha, glucosa_ayunas, glucosa_2h, a1c, ldl, hdl, trigliceridos, colesterol_total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (current_patient_id, dia, fecha, glucosa_ayunas or None, glucosa_2h or None, a1c or None, ldl or None, hdl or None, trigliceridos or None, colesterol_total or None))
    db.commit()
    flash('Lectura guardada correctamente.')
    return redirect(url_for('professional_dashboard'))
 
@app.route('/save_note', methods=['POST'])
def save_note():
    if 'user_id' not in session or session['user_type'] != 'profesional':
        return redirect(url_for('login'))
 
    current_patient_id = session.get('current_patient_id')
    if not current_patient_id:
        flash('Selecciona un paciente primero.')
        return redirect(url_for('professional_dashboard'))
 
    nota = request.form.get('nota', '').strip()
    if not nota:
        flash('La nota no puede estar vacía.')
        return redirect(url_for('professional_dashboard'))
 
    fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS notas_paciente (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER,
            nota TEXT,
            fecha TEXT,
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        )
    ''')
    db.execute('INSERT INTO notas_paciente (paciente_id, nota, fecha) VALUES (?, ?, ?)', (current_patient_id, nota, fecha))
    db.commit()
    flash('Nota guardada correctamente.')
    return redirect(url_for('professional_dashboard'))
 
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
 
@app.route('/view_patient/<int:paciente_id>')
def view_patient(paciente_id):
    if 'user_id' not in session or session['user_type'] != 'profesional':
        return redirect(url_for('login'))
    db = get_db()
    paciente = db.execute('SELECT * FROM pacientes WHERE id = ?', (paciente_id,)).fetchone()
    if not paciente:
        flash('Paciente no encontrado.')
        return redirect(url_for('professional_dashboard'))
 
    lecturas = db.execute('SELECT * FROM lecturas WHERE paciente_id = ? ORDER BY fecha DESC', (paciente_id,)).fetchall()
    return render_template('view_patient.html', paciente=paciente, lecturas=lecturas)
 
@app.route('/evaluate_patient/<int:paciente_id>')
def evaluate_patient(paciente_id):
    if 'user_id' not in session or session['user_type'] != 'profesional':
        return redirect(url_for('login'))
    db = get_db()
    paciente = db.execute('SELECT * FROM pacientes WHERE id = ?', (paciente_id,)).fetchone()
    ultima_lectura = db.execute('SELECT * FROM lecturas WHERE paciente_id = ? ORDER BY fecha DESC LIMIT 1', (paciente_id,)).fetchone()
    if ultima_lectura and diabetes_model:
        features = [
            paciente['edad'],
            ultima_lectura['glucosa_ayunas'] or 0,
            ultima_lectura['glucosa_2h'] or 0,
            ultima_lectura['a1c'] or 0,
            25,
            ultima_lectura['ldl'] or 0,
            ultima_lectura['hdl'] or 0,
            ultima_lectura['trigliceridos'] or 0,
            ultima_lectura['colesterol_total'] or 0,
            120, 80, 5, 0, 150, 7
        ]
        try:
            prob, stage, risk_factors, recommendations = assess_diabetes_risk_comprehensive(features, diabetes_model)
            return render_template('evaluation.html', paciente=paciente, prob=prob, stage=stage, risk_factors=risk_factors, recommendations=recommendations)
        except Exception as e:
            flash(f'Error en evaluación: {str(e)}')
    return render_template('evaluation.html', paciente=paciente, error="No hay suficientes datos o modelo no disponible")
 
 
if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)