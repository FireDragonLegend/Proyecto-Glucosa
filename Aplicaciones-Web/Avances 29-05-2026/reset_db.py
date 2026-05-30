import os
import sqlite3

# Eliminar DB vieja
db_path = "app.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✓ {db_path} eliminada")

# Crear DB nueva
conn = sqlite3.connect(db_path)
with open('schema.sql', 'r') as f:
    conn.executescript(f.read())
conn.commit()
conn.close()
print("✓ Nueva base de datos creada con el nuevo schema")
