from werkzeug.security import generate_password_hash
import sqlite3

conn = sqlite3.connect('app.db')
cur = conn.cursor()
new_hash = generate_password_hash('test123')
cur.execute('UPDATE profesionales SET password = ? WHERE id = ?', (new_hash, 1))
conn.commit()
conn.close()
print('Password updated for professional id=1 (test123)')
