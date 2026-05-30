-- Tabla para pacientes
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    fecha_nacimiento DATE NOT NULL,
    lugar_tratamiento TEXT,
    nacionalidad TEXT,
    edad INTEGER,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    fecha_ingreso DATE NOT NULL,
    doctor_asignado TEXT DEFAULT 'No asignado'
);

-- Tabla para profesionales
CREATE TABLE IF NOT EXISTS profesionales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cedula TEXT UNIQUE NOT NULL,
    nombre TEXT NOT NULL,
    edad INTEGER,
    fecha_nacimiento DATE NOT NULL,
    lugar_trabajo TEXT,
    nacionalidad TEXT,
    curp TEXT,
    rfc TEXT,
    clues TEXT,
    cct TEXT,
    password TEXT NOT NULL,
    foto_path TEXT,
    fecha_ingreso DATE NOT NULL
);

-- Tabla para lecturas de glucosa (similar a Proyecto.db)
CREATE TABLE IF NOT EXISTS lecturas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER,
    dia INTEGER,
    fecha DATE,
    glucosa_ayunas REAL,
    glucosa_2h REAL,
    a1c REAL,
    ldl REAL,
    hdl REAL,
    trigliceridos REAL,
    colesterol_total REAL,
    FOREIGN KEY (paciente_id) REFERENCES pacientes (id)
);

-- Tabla para participantes (si es necesario)
CREATE TABLE IF NOT EXISTS participantes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    edad INTEGER,
    fecha DATE,
    lugar TEXT,
    fecha_nac DATE,
    pdf_path TEXT
);