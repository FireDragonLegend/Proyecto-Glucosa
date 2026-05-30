#!/usr/bin/env python
# Script de prueba para inicializar la BD

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    with open('schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print('✓ Base de datos inicializada correctamente')

if __name__ == '__main__':
    init_db()
