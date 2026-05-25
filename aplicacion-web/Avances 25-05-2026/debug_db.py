import sqlite3, os

path = 'app.db'
print('exists', os.path.exists(path))
conn = sqlite3.connect(path)
cur = conn.cursor()
print('tables', [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")])
print('pacientes', list(cur.execute('SELECT id,nombre,email FROM pacientes')))
print('profesionales', list(cur.execute('SELECT id,cedula,nombre FROM profesionales')))
conn.close()
