import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import statistics
import csv
from datetime import datetime
import sqlite3
import threading
from diabetes_model import (
    assess_diabetes_risk_comprehensive,
    map_probability_to_timeframe,
    possible_symptoms_by_probability,
    classify_lipid_profile,
    generate_lipid_recommendations,
    AdvancedDiabetesModel,
)
import pickle

class GlucoseMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor de Glucosa - 3 meses")
        self.root.geometry("760x520")
        # lista de dicts: {'day':int,'date':str,'fast':float,'post':float,'a1c':float or None}
        self.readings = []

        self.next_day = 1
        self.default_a1c = None

        # Database
        self.db_path = "Proyecto.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.current_participant_id = None  # id del participante seleccionado
        self.crear_tablas_db()
        self.participantes = []  # lista de tuples (id, nombre)
        self.cargar_participantes_desde_db()

        # modelo IA avanzado (se carga en background para no bloquear la interfaz)
        self.diabetes_model = None  # referencia al modelo
        self.model_loading = False  # indicador de carga

        self._build_ui()

        # iniciar carga del modelo en segundo plano
        # inicia un hilo para cargar/entrenar modelo sin bloquear UI
        threading.Thread(target=self._background_load_model, daemon=True).start()

    def _background_load_model(self):
        """Método interno ejecutado en segundo plano para preparar el modelo IA."""
        self.model_loading = True
        self.diabetes_model = self.load_or_train_model()
        self.model_loading = False

    def load_or_train_model(self):
        """Carga el modelo entrenado desde archivo, o lo entrena si no existe."""
        import os
        model_path = "diabetes_model.pkl"

        print(f"Buscando modelo en: {os.path.abspath(model_path)}")

        if os.path.exists(model_path):
            print("Archivo de modelo encontrado, intentando cargar...")
            try:
                # Intentar cargar modelo guardado
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("✓ Modelo de IA cargado exitosamente desde archivo.")
                return model
            except Exception as e:
                print(f"Error al cargar modelo existente: {e}")
        else:
            print("Archivo de modelo no encontrado, se entrenará uno nuevo.")

        # Si no existe o está corrupto, entrenar nuevo modelo
        print("Entrenando nuevo modelo de IA (versión optimizada)...")
        try:
            # Importar solo cuando sea necesario
            from diabetes_model import generate_advanced_dataset

            print("Generando dataset de entrenamiento optimizado...")
            X, y = generate_advanced_dataset(1000)  # Dataset optimizado para producción
            print(f"Dataset generado: {len(X)} muestras")

            print("Entrenando modelo de IA optimizado...")
            model = AdvancedDiabetesModel(n_features=15)
            model.fit(X, y, epochs=200, batch_size=64, validation_split=0.2, patience=15)  # Entrenamiento optimizado
            print("Modelo entrenado exitosamente")

            # Guardar el modelo entrenado
            print(f"Guardando modelo en: {os.path.abspath(model_path)}")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print("✓ Modelo entrenado y guardado exitosamente.")
            return model
        except Exception as e:
            print(f"✗ Error al entrenar modelo: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error de IA",
                "No se pudo cargar ni entrenar el modelo de IA.\n"
                "El sistema funcionará sin predicciones avanzadas.")
            return None

    def get_diabetes_model(self):
        """Obtiene el modelo de IA, cargándolo si es necesario."""
        if self.diabetes_model is None:
            if self.model_loading:
                messagebox.showinfo("IA en proceso", "El modelo de IA todavía se está cargando. Por favor, inténtelo de nuevo en un momento.")
                return None
            self.diabetes_model = self.load_or_train_model()
        return self.diabetes_model

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        # colores base: labels con combinación de azul y negro, fondo blanco
        style.configure("TLabel", font=("Segoe UI", 10), background="white", foreground="#1f2937")
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="white", background="white")
        style.configure("TFrame", background="white")
        # Header con fondo blanco y letras en azul oscuro
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="white", foreground="#1e40af")
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=24, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", background="#f8fafc", foreground="#1e40af", font=("Segoe UI", 9, "bold"))
        # contenedores blancos tipo tarjeta sin colores
        style.configure("Card.TFrame", background="white", relief="raised", borderwidth=1)
        # estilos para LabelFrame tipo tarjeta - títulos en NEGRO, fondo blanco
        style.configure("TLabelframe", background="white", foreground="#1f2937")
        style.configure("TLabelframe.Label", background="white", foreground="#000000", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[('selected', '#bfdbfe')], foreground=[('selected', '#000000')])
        # botones intuitivos: agregar verde brillante, eliminar rojo brillante
        style.configure("Primary.TButton", foreground="white", background="#10b981", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#059669')])
        style.configure("Danger.TButton", foreground="white", background="#ef4444", font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#dc2626')])

        self.root.geometry("1300x850")
        self.root.minsize(1200, 750)
        try:
            self.root.configure(bg="white")  # fondo blanco
        except:
            pass

        # encabezado principal tipo banner
        banner = ttk.Label(self.root, text="📊 Monitor de Glucosa - Clínica", style="Header.TLabel")
        banner.pack(fill="x", pady=(3, 10))

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.configure(style="TFrame")
        paned.pack(fill="both", expand=True, padx=12, pady=12)

        # Contenedor izquierdo scrollable: usamos un Canvas + scrollbar
        left_container = ttk.Frame(paned, width=450)
        left_container.configure(style="Card.TFrame")
        paned.add(left_container, weight=0)

        canvas_left = tk.Canvas(left_container, highlightthickness=0)
        vscroll = ttk.Scrollbar(left_container, orient="vertical", command=canvas_left.yview)
        canvas_left.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas_left.pack(side="left", fill="both", expand=True)

        # frame real donde añadiremos widgets (mantendremos nombre 'left')
        left = ttk.Frame(canvas_left, width=430)
        left.configure(style="Card.TFrame")
        canvas_left.create_window((0, 0), window=left, anchor="nw")

        def _left_configure(event):
            canvas_left.configure(scrollregion=canvas_left.bbox("all"))

        left.bind("<Configure>", _left_configure)

        # Nuevo: panel de registro/selección de participante
        frm_part = ttk.LabelFrame(left, text="Participante / Proyecto", padding=12)
        frm_part.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_part, text="Seleccionar participante:").grid(row=0, column=0, sticky="w")
        self.participante_var = tk.StringVar()
        self.participante_cb = ttk.Combobox(frm_part, textvariable=self.participante_var, state="readonly", width=35)
        self.participante_cb.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        self.participante_cb.bind("<<ComboboxSelected>>", lambda e: self.on_participante_selected())

        ttk.Separator(frm_part, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)

        ttk.Label(frm_part, text="Nombre:").grid(row=2, column=0, sticky="w")
        self.new_nombre = tk.StringVar()
        ttk.Entry(frm_part, textvariable=self.new_nombre, width=20).grid(row=2, column=1, sticky="w")

        ttk.Label(frm_part, text="Edad:").grid(row=3, column=0, sticky="w")
        self.new_edad = tk.StringVar()
        ttk.Entry(frm_part, textvariable=self.new_edad, width=8).grid(row=3, column=1, sticky="w")

        ttk.Label(frm_part, text="Fecha (YYYY-MM-DD):").grid(row=4, column=0, sticky="w")
        self.new_fecha = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(frm_part, textvariable=self.new_fecha, width=12).grid(row=4, column=1, sticky="w")

        ttk.Label(frm_part, text="Lugar:").grid(row=5, column=0, sticky="w")
        self.new_lugar = tk.StringVar()
        ttk.Entry(frm_part, textvariable=self.new_lugar, width=20).grid(row=5, column=1, sticky="w")

        btn_reg = ttk.Button(frm_part, text="Registrar participante", command=lambda: self.registrar_participante_db(self.new_nombre.get(), self.new_edad.get(), self.new_fecha.get(), self.new_lugar.get()), style="Primary.TButton")
        btn_reg.grid(row=6, column=0, columnspan=2, pady=6, sticky="we")

        frm_inputs = ttk.LabelFrame(left, text="Agregar lectura diaria", padding=12)
        frm_inputs.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_inputs, text="Día (1-90):", style="TLabel").grid(row=0, column=0, sticky="w", pady=4)
        self.day_var = tk.IntVar(value=self.next_day)
        self.ent_day = ttk.Entry(frm_inputs, textvariable=self.day_var, width=8, font=("Segoe UI", 10))
        self.ent_day.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_inputs, text="Glucosa en ayunas (mg/dL):").grid(row=1, column=0, sticky="w", pady=4)
        self.fast_var = tk.StringVar()
        self.ent_fast = ttk.Entry(frm_inputs, textvariable=self.fast_var, width=12, font=("Segoe UI", 10))
        self.ent_fast.grid(row=1, column=1, padx=6, pady=4, sticky="w")
        ttk.Label(frm_inputs, text="(mínimo: 15)", font=("Segoe UI", 8), foreground="#6366f1").grid(row=1, column=2, sticky="w", padx=2)

        ttk.Label(frm_inputs, text="Glucosa 2h pos comida (mg/dL):").grid(row=2, column=0, sticky="w", pady=4)
        self.post_var = tk.StringVar()
        self.ent_post = ttk.Entry(frm_inputs, textvariable=self.post_var, width=12, font=("Segoe UI", 10))
        self.ent_post.grid(row=2, column=1, padx=6, pady=4, sticky="w")
        ttk.Label(frm_inputs, text="(mínimo: 40, hipoglucemia si ≤70)", font=("Segoe UI", 8), foreground="#6366f1").grid(row=2, column=2, sticky="w", padx=2)

        ttk.Label(frm_inputs, text="A1C (% - opcional):").grid(row=3, column=0, sticky="w", pady=4)
        self.a1c_var = tk.StringVar()
        self.ent_a1c = ttk.Entry(frm_inputs, textvariable=self.a1c_var, width=12, font=("Segoe UI", 10))
        self.ent_a1c.grid(row=3, column=1, padx=6, pady=4, sticky="w")
        ttk.Label(frm_inputs, text="(mínimo: 3.5)", font=("Segoe UI", 8), foreground="#6366f1").grid(row=3, column=2, sticky="w", padx=2)

        btn_frame = ttk.Frame(frm_inputs)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=(8, 0), sticky="we")
        btn_frame.columnconfigure((0, 1), weight=1)

        btn_add = ttk.Button(btn_frame, text="Agregar lectura", command=self.add_reading, style="Primary.TButton")
        btn_add.grid(row=0, column=0, padx=(0,6), sticky="we")

        btn_reset = ttk.Button(btn_frame, text="Resetear semana", command=self.reset_week, style="Danger.TButton")
        btn_reset.grid(row=0, column=1, sticky="we")

        util_frame = ttk.Frame(frm_inputs)
        util_frame.grid(row=5, column=0, columnspan=3, pady=6, sticky="we")
        util_frame.columnconfigure((0,1), weight=1)

        btn_export = ttk.Button(util_frame, text="Exportar CSV", command=self.export_csv)
        btn_export.grid(row=0, column=0, padx=(0,6), sticky="we")
        btn_history = ttk.Button(util_frame, text="Historial detallado", command=self.show_history_window)
        btn_history.grid(row=0, column=1, sticky="we")

        frm_summary = ttk.LabelFrame(left, text="Resumen rápido (para profesional)", padding=12)
        frm_summary.pack(fill="both", expand=False, padx=10, pady=8)

        self.lbl_summary = ttk.Label(frm_summary, text="No hay lecturas aún.", wraplength=350, justify="left")
        self.lbl_summary.pack(fill="both", expand=True, padx=6, pady=6)

        btn_full_summary = ttk.Button(frm_summary, text="Resumen semanal completo", command=self.show_summary)
        btn_full_summary.pack(fill="x", pady=6)

        frm_expert = ttk.LabelFrame(left, text="Herramientas para profesional", padding=12)
        frm_expert.pack(fill="both", expand=False, padx=10, pady=8)

        btn_eval_sel = ttk.Button(frm_expert, text="Evaluar lectura seleccionada", command=self.show_selected_assessment)
        btn_eval_sel.pack(fill="x", pady=2)
        btn_eval_ai = ttk.Button(frm_expert, text="Evaluar riesgo IA Avanzada", command=self.evaluate_diabetes_risk)
        btn_eval_ai.pack(fill="x", pady=2)
        btn_train_db = ttk.Button(frm_expert, text="Entrenar IA con BD", command=self.train_model_from_db)
        btn_train_db.pack(fill="x", pady=2)
        btn_show_options = ttk.Button(frm_expert, text="Ver opciones", command=self.show_app_options)
        btn_show_options.pack(fill="x", pady=2)

        right = ttk.Frame(paned)
        right.configure(style="Card.TFrame")
        paned.add(right, weight=1)

        header = ttk.Label(right, text="Historial de la semana", style="Header.TLabel")
        header.pack(anchor="w", padx=10, pady=(12, 6))

        cols = ("día", "fecha", "ayuno (mg/dL)", "2h pos (mg/dL)", "A1C (%)", "clasif ayuno", "clasif 2h", "clasif A1C")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
        # alternado de filas para facilitar lectura
        self.tree.tag_configure('oddrow', background='#f9fafb')
        self.tree.tag_configure('evenrow', background='white')
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("día", width=55, anchor="center")
        self.tree.column("fecha", width=95, anchor="center")
        self.tree.column("ayuno (mg/dL)", width=110, anchor="center")
        self.tree.column("2h pos (mg/dL)", width=120, anchor="center")
        self.tree.column("A1C (%)", width=75, anchor="center")
        self.tree.column("clasif ayuno", width=120, anchor="center")
        self.tree.column("clasif 2h", width=120, anchor="center")
        self.tree.column("clasif A1C", width=120, anchor="center")

        self.tree.pack(fill="both", expand=True, padx=10, pady=8)
        # Ahora que el tree ya existe, inicializar valores del combobox y cargar lecturas
        self._refresh_participante_combobox()

        frm_detail = ttk.LabelFrame(right, text="Detalle y evaluación para profesional", padding=12)
        frm_detail.pack(fill="x", padx=10, pady=8)

        self.txt_detail = tk.Text(frm_detail, height=8, wrap="word", font=("Segoe UI", 10), relief="flat", bg="#ffffff")
        self.txt_detail.pack(fill="both", expand=True, padx=6, pady=6)

        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select())
        for widget in (self.ent_day, self.ent_fast, self.ent_post, self.ent_a1c):
            widget.bind("<Return>", lambda e: self._on_enter_from_entry())

    def crear_tablas_db(self):
        """Crear tablas necesarias: participantes y lecturas (con colesterol)"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS participantes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                edad INTEGER,
                fecha TEXT,
                lugar TEXT
            )
        """)
        # lecturas: incluye glucosa y colesterol
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS lecturas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participante_id INTEGER,
                dia INTEGER,
                fecha TEXT,
                ayuno REAL,
                pos2h REAL,
                a1c REAL,
                ldl REAL,
                hdl REAL,
                triglycerides REAL,
                total_cholesterol REAL,
                UNIQUE(participante_id, dia),
                FOREIGN KEY (participante_id) REFERENCES participantes(id)
            )
        """)
        # Intentar agregar columnas si faltan (para compatibilidad con BD antiguas)
        try:
            self.cursor.execute("ALTER TABLE lecturas ADD COLUMN ldl REAL")
        except sqlite3.OperationalError:
            pass  # La columna ya existe
        try:
            self.cursor.execute("ALTER TABLE lecturas ADD COLUMN hdl REAL")
        except sqlite3.OperationalError:
            pass
        try:
            self.cursor.execute("ALTER TABLE lecturas ADD COLUMN triglycerides REAL")
        except sqlite3.OperationalError:
            pass
        try:
            self.cursor.execute("ALTER TABLE lecturas ADD COLUMN total_cholesterol REAL")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def cargar_participantes_desde_db(self):
        """Carga participantes desde BD para el combobox"""
        self.participantes.clear()
        self.cursor.execute("SELECT id, nombre FROM participantes ORDER BY nombre")
        filas = self.cursor.fetchall()
        for fila in filas:
            self.participantes.append((fila[0], fila[1]))

    def cargar_lecturas_desde_db(self, participante_id):
        """Carga lecturas de un participante específico"""
        self.readings.clear()
        if participante_id is None:
            return
        self.cursor.execute("""
            SELECT dia, fecha, ayuno, pos2h, a1c
            FROM lecturas
            WHERE participante_id = ?
            ORDER BY dia
        """, (participante_id,))
        filas = self.cursor.fetchall()
        for f in filas:
            entry = {
                'day': f[0],
                'date': f[1],
                'fast': f[2] if f[2] is not None else 0.0,
                'post': f[3] if f[3] is not None else 0.0,
                'a1c': f[4]
            }
            self.readings.append(entry)
        self.readings.sort(key=lambda x: x['day'])
        # actualizar default_a1c si hay alguna lecturas con a1c
        a1c_vals = [r['a1c'] for r in self.readings if r['a1c'] is not None]
        if a1c_vals:
            self.default_a1c = a1c_vals[-1]
        else:
            self.default_a1c = None
        # después de cargar lecturas, ajustar siguiente día y prefills si la UI ya existe
        if hasattr(self, "day_var"):
            self._set_next_day_and_prefill_inputs()

    def registrar_participante_db(self, nombre, edad, fecha, lugar):
        """Inserta un participante en BD y recarga combobox"""
        if not nombre.strip():
            messagebox.showerror("Error", "El nombre no puede estar vacío.")
            return
        try:
            edad_val = int(edad) if str(edad).strip() != "" else None
        except:
            messagebox.showerror("Error", "Edad inválida.")
            return
        self.cursor.execute(
            "INSERT INTO participantes (nombre, edad, fecha, lugar) VALUES (?, ?, ?, ?)",
            (nombre.strip(), edad_val, fecha.strip(), lugar.strip())
        )
        self.conn.commit()
        self.cargar_participantes_desde_db()
        self._refresh_participante_combobox()
        messagebox.showinfo("Participante", f"✓ Participante '{nombre}' registrado.")

    def guardar_lectura_en_db(self, participante_id, entry):
        """Inserta o actualiza (upsert) una lectura del participante con opción de colesterol"""
        if participante_id is None:
            return
        # usar INSERT OR REPLACE aprovechando UNIQUE(participante_id,dia)
        # Los campos de colesterol pueden ser None si no se han ingresado
        ldl = entry.get('ldl', None)
        hdl = entry.get('hdl', None)
        triglycerides = entry.get('triglycerides', None)
        total_cholesterol = entry.get('total_cholesterol', None)
        
        self.cursor.execute("""
            INSERT INTO lecturas (participante_id, dia, fecha, ayuno, pos2h, a1c, ldl, hdl, triglycerides, total_cholesterol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participante_id, dia) DO UPDATE SET
                fecha=excluded.fecha,
                ayuno=excluded.ayuno,
                pos2h=excluded.pos2h,
                a1c=excluded.a1c,
                ldl=excluded.ldl,
                hdl=excluded.hdl,
                triglycerides=excluded.triglycerides,
                total_cholesterol=excluded.total_cholesterol
        """, (
            participante_id,
            entry['day'],
            entry['date'],
            entry['fast'],
            entry['post'],
            entry['a1c'],
            ldl,
            hdl,
            triglycerides,
            total_cholesterol
        ))
        self.conn.commit()

    def _refresh_participante_combobox(self):
        names = [p[1] for p in self.participantes]
        self.participante_cb['values'] = names
        # si hay participantes y ninguno seleccionado, seleccionar el primero
        if names and self.current_participant_id is None:
            self.participante_cb.set(names[0])
            # seleccionar el id correspondiente
            first_id = self.participantes[0][0]
            self.current_participant_id = first_id
            self.cargar_lecturas_desde_db(self.current_participant_id)
            # refrescar árbol solo si existe (evita errores de orden de inicialización)
            if hasattr(self, "tree"):
                self._refresh_tree()
                # también ajustar inputs según lecturas cargadas
                self._set_next_day_and_prefill_inputs()

    def on_participante_selected(self):
        sel_name = self.participante_var.get()
        if not sel_name:
            return
        pid = next((p[0] for p in self.participantes if p[1] == sel_name), None)
        self.current_participant_id = pid
        self.cargar_lecturas_desde_db(pid)
        self._refresh_tree()
        # ajustar inputs (día siguiente y A1C por defecto) al cambiar participante
        self._set_next_day_and_prefill_inputs()

    def _on_enter_from_entry(self):
        """
        Llamado cuando el usuario presiona Enter en un Entry.
        Si los 3 campos numéricos están con valor, invoca add_reading().
        """
        # si algunos campos vacíos, no hacer nada (permite edición normal)
        vday = str(self.day_var.get()).strip()
        vfast = self.fast_var.get().strip()
        vpost = self.post_var.get().strip()
        # A1C puede estar vacío (opcional) pero si está vacío igual se añadirá
        # Según petición, permitir captura por Enter cuando "todos los datos ya puestos"
        if vday != "" and vfast != "" and vpost != "":
            # intentar parsear para validar
            try:
                int(vday)
                float(vfast)
                float(vpost)
            except:
                # no validar con diálogo intrusivo aquí; dejar que add_reading muestre error
                pass
            # llamar a add_reading (gestiona errores y mensajes)
            self.add_reading()

    def _on_tree_select(self):
        sel = self.tree.selection()
        if not sel:
            self.txt_detail.delete("1.0", tk.END)
            return
        vals = self.tree.item(sel[0], "values")
        # buscar entry por día
        try:
            day = int(vals[0])
        except:
            return
        entry = next((r for r in self.readings if r['day'] == day), None)
        if not entry:
            return
        # generar evaluación breve (sin mostrar diálogo)
        severity, resumen = self.format_assessment_for_professional(entry)
        self.txt_detail.delete("1.0", tk.END)
        header = f"Lectura día {entry['day']} - Severidad: {severity}\n\n"
        self.txt_detail.insert(tk.END, header + resumen)

    # Clasificadores según tus umbrales
    def classify_fast(self, v):
        if v is None:
            return "-"
        if v < 15:
            return "Valor inválido (< 15)"
        if v < 60:
            return "Hipoglucemia"
        if 60 <= v <= 100:
            return "Normal"
        if 100 < v <= 125:
            return "Prediabetes"
        return "Diabetes"

    def classify_post(self, v):
        if v is None:
            return "-"
        if v < 40:
            return "Valor inválido (< 40)"
        if v <= 70:
            return "Hipoglucemia"
        if 70 < v <= 120:
            return "Normal"
        if 120 < v <= 140:
            return "Límite alto"
        if 141 <= v <= 199:
            return "Prediabetes"
        return "Diabetes"

    def classify_a1c(self, v):
        if v is None:
            return "-"
        if v < 3.5:
            return "Valor inválido (< 3.5%)"
        if v < 5.0:
            return "Normal"
        if 5.6 <= v <= 6.4:
            return "Prediabetes"
        if v >= 6.5:
            return "Diabetes"
        # entre 5.0 y 5.5 es considerado en rango intermedio
        return "Rango intermedio"

    def add_reading(self):
        try:
            day = int(self.day_var.get())
            if not (1 <= day <= 90):
                raise ValueError("Día debe estar entre 1 y 90")
        except Exception as e:
            messagebox.showerror("Error", f"Día inválido: {e}")
            return

        fast = None
        post = None
        if self.fast_var.get().strip() != "":
            try:
                fast = float(self.fast_var.get())
            except:
                messagebox.showerror("Error", "Glucosa en ayunas inválida")
                return
        if self.post_var.get().strip() != "":
            try:
                post = float(self.post_var.get())
            except:
                messagebox.showerror("Error", "Glucosa 2h posprandial inválida")
                return

        a1c = None
        if self.a1c_var.get().strip() != "":
            try:
                a1c = float(self.a1c_var.get())
            except:
                messagebox.showerror("Error", "A1C inválido")
                return

        # VALIDACIONES DE VALORES MÍNIMOS
        error_msg = ""
        if fast is not None and fast < 15:
            error_msg += f"• Glucosa en ayunas: {fast:.1f} mg/dL (mínimo válido: 15 mg/dL)\n"
        if post is not None and post < 40:
            error_msg += f"• Glucosa 2h después de comer: {post:.1f} mg/dL (mínimo válido: 40 mg/dL)\n"
        if a1c is not None and a1c < 3.5:
            error_msg += f"• A1C: {a1c:.2f}% (mínimo válido: 3.5%)\n"
        
        if error_msg:
            messagebox.showerror(
                "⚠️ Valor No Válido",
                f"Los siguientes valores están fuera del rango permitido:\n\n{error_msg}\n"
                f"Por favor, verifique los datos ingresados.\nEl día NO será registrado hasta que corrija estos valores."
            )
            return

        existing = next((r for r in self.readings if r['day'] == day), None)
        if existing:
            # Si solo se actualiza A1C y no los valores, propagar
            if a1c is not None and self.fast_var.get().strip() == "" and self.post_var.get().strip() == "":
                self.default_a1c = a1c
                for r in self.readings:
                    r['a1c'] = a1c
                # actualizar BD para participante
                if self.current_participant_id:
                    for r in self.readings:
                        self.guardar_lectura_en_db(self.current_participant_id, r)
                self._refresh_tree()
                messagebox.showinfo("A1C actualizada", f"A1C actualizada a {a1c:.2f}% en todos los días existentes y se usará por defecto.")
                self.a1c_var.set("")
                return
            else:
                # si ya existe y están llenos otros campos, actualizar esa lectura
                existing['fast'] = fast if fast is not None else existing['fast']
                existing['post'] = post if post is not None else existing['post']
                existing['a1c'] = a1c if a1c is not None else existing['a1c']
                existing['date'] = datetime.now().strftime("%Y-%m-%d")
                # guardar en BD
                if self.current_participant_id:
                    self.guardar_lectura_en_db(self.current_participant_id, existing)
                self._refresh_tree()
                messagebox.showinfo("Actualizado", f"Lectura día {day} actualizada.")
                self._clear_inputs_after_add()
                return

        if a1c is None and self.default_a1c is not None:
            a1c = self.default_a1c

        entry = {
            'day': day,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'fast': fast if fast is not None else 0.0,
            'post': post if post is not None else 0.0,
            'a1c': a1c
        }
        self.readings.append(entry)
        self.readings.sort(key=lambda x: x['day'])

        if a1c is not None:
            self.default_a1c = a1c
            for r in self.readings:
                r['a1c'] = a1c

        # guardar en BD si hay participante seleccionado
        if self.current_participant_id:
            self.guardar_lectura_en_db(self.current_participant_id, entry)
        else:
            messagebox.showwarning("Sin participante", "No hay participante seleccionado. La lectura se guarda solo en la sesión hasta seleccionar un participante.")

        self._refresh_tree()
        self.assess_with_history(entry)

        self.next_day = min(90, max(1, self.next_day + 1))
        self.day_var.set(self.next_day)
        self._clear_inputs_after_add()

    def _clear_inputs_after_add(self):
        self.fast_var.set("")
        self.post_var.set("")
        self.a1c_var.set("")

    def _refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, r in enumerate(self.readings):
            cf = self.classify_fast(r['fast'])
            cp = self.classify_post(r['post'])
            ca = self.classify_a1c(r['a1c']) if r['a1c'] is not None else "-"
            a1c_display = f"{r['a1c']:.2f}" if r['a1c'] is not None else "-"
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            self.tree.insert("", "end", values=(r['day'], r['date'], f"{r['fast']:.1f}", f"{r['post']:.1f}", a1c_display, cf, cp, ca), tags=(tag,))

    def show_summary(self):
        if not self.readings:
            messagebox.showinfo("Resumen", "No hay lecturas para la semana.")
            return
        fast_vals = [r['fast'] for r in self.readings]
        post_vals = [r['post'] for r in self.readings]
        a1c_vals = [r['a1c'] for r in self.readings if r['a1c'] is not None]

        avg_fast = statistics.mean(fast_vals)
        avg_post = statistics.mean(post_vals)
        avg_a1c = statistics.mean(a1c_vals) if a1c_vals else None

        cf = self.classify_fast(avg_fast)
        cp = self.classify_post(avg_post)
        ca = self.classify_a1c(avg_a1c) if avg_a1c is not None else "-"

        # heurística simple de resistencia a la insulina (orientativa):
        insulin_resistance_hint = "No suficiente info"
        if (100 < avg_fast <= 125) and (avg_a1c is not None and avg_a1c >= 5.6):
            insulin_resistance_hint = "Posible resistencia a la insulina (orientativo)"
        elif avg_fast > 125 or avg_post > 199 or (avg_a1c is not None and avg_a1c >= 6.5):
            insulin_resistance_hint = "Valores consistentes con diabetes - consulte a un profesional"
        else:
            insulin_resistance_hint = "No hay indicios claros de resistencia según estos datos"

        txt = (
            "Para su profesional de la salud:\n\n"
            f"Promedios del periodo (n={len(self.readings)} lecturas, hasta 90 días):\n\n"
            f"Glucosa en ayunas promedio: {avg_fast:.1f} mg/dL -> {cf}\n"
            f"Glucosa 2h posprandial promedio: {avg_post:.1f} mg/dL -> {cp}\n"
        )
        if avg_a1c is not None:
            txt += f"A1C promedio: {avg_a1c:.2f}% -> {ca}\n"
        else:
            txt += "A1C promedio: no disponible\n"
        txt += f"\nInterpretación de resistencia a la insulina: {insulin_resistance_hint}\n\n"
        txt += "Nota: Información orientativa dirigida al profesional de la salud. Diagnóstico por profesional."

        # además de mostrar diálogo, actualizar resumen rápido en panel izquierdo
        self.lbl_summary.config(text=txt.replace("Para su profesional de la salud:\n\n", ""))
        messagebox.showinfo("Resumen semanal (para profesional)", txt)

    def reset_week(self):
        if messagebox.askyesno("Confirmar", "¿Desea borrar todas las lecturas del periodo (hasta 90 días) (solo en BD para el participante seleccionado)?"):
            # borra lecturas en sesión y en BD para participante seleccionado
            if self.current_participant_id:
                self.cursor.execute("DELETE FROM lecturas WHERE participante_id = ?", (self.current_participant_id,))
                self.conn.commit()
            self.readings = []
            self._refresh_tree()
            self.next_day = 1
            self.day_var.set(self.next_day)

    def export_csv(self):
        if not self.readings:
            messagebox.showinfo("Exportar", "No hay lecturas para exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], initialfile="glucosa_semana.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["día", "fecha", "ayuno_mg/dL", "2h_pos_mg/dL", "A1C_%"])
            for r in self.readings:
                writer.writerow([r['day'], r['date'], r['fast'], r['post'], r['a1c'] if r['a1c'] is not None else ""])
        messagebox.showinfo("Exportar", f"Exportado a {path}")

    def format_assessment_for_professional(self, new_entry):
        """
        Devuelve (severidad, resumen_string) sin dialogos.
        Severidad: 'Crítica', 'Advertencia', 'Seguimiento', 'Normal'
        """
        prev = [r for r in self.readings if r['day'] != new_entry['day']]
        if not prev:
            # sin contexto previo -> usar evaluación basada solo en la lectura actual
            cf_new = self.classify_fast(new_entry['fast'])
            cp_new = self.classify_post(new_entry['post'])
            ca_new = self.classify_a1c(new_entry['a1c']) if new_entry['a1c'] is not None else "-"
            resumen = (
                f"Ayuno {new_entry['fast']:.1f} mg/dL ({cf_new}), 2h pos {new_entry['post']:.1f} mg/dL ({cp_new}), "
                f"A1C: {str(new_entry['a1c']) if new_entry['a1c'] is not None else 'no disponible'} ({ca_new})."
            )
            # determinar severidad por valores absolutos
            if (new_entry['fast'] > 125) or (new_entry['post'] > 199) or (new_entry['a1c'] is not None and new_entry['a1c'] >= 6.5):
                return "Crítica", resumen + "\nPosible DIABETES (orientativo). Evaluación clínica prioritaria."
            if (100 < new_entry['fast'] <= 125) or (141 <= new_entry['post'] <= 199) or (new_entry['a1c'] is not None and 5.6 <= new_entry['a1c'] <= 6.4):
                return "Advertencia", resumen + "\nIndicios de PREDIABETES (orientativo). Seguimiento recomendado."
            return "Normal", resumen + "\nLectura dentro de rangos esperados."

        # con contexto previo
        prev_fast_vals = [r['fast'] for r in prev]
        prev_post_vals = [r['post'] for r in prev]
        prev_a1c_vals = [r['a1c'] for r in prev if r['a1c'] is not None]
        avg_prev_fast = statistics.mean(prev_fast_vals)
        avg_prev_post = statistics.mean(prev_post_vals)
        avg_prev_a1c = statistics.mean(prev_a1c_vals) if prev_a1c_vals else None

        cf_new = self.classify_fast(new_entry['fast'])
        cp_new = self.classify_post(new_entry['post'])
        ca_new = self.classify_a1c(new_entry['a1c']) if new_entry['a1c'] is not None else "-"

        mensajes = []
        severidad = "Normal"

        # chequeo crítico
        if (new_entry['fast'] > 125) or (new_entry['post'] > 199) or (new_entry['a1c'] is not None and new_entry['a1c'] >= 6.5):
            mensajes.append("Posible DIABETES (orientativo). Sugerir evaluación clínica prioritaria.")
            severidad = "Crítica"

        # empeoramiento respecto al promedio previo:
        # antes se avisaba por aumento absoluto; ahora solo se alerta si además
        # las 3 medidas de la nueva lectura están en rango de PREDIABETES
        delta_fast = new_entry['fast'] - avg_prev_fast
        delta_post = new_entry['post'] - avg_prev_post
        increases_significant = (delta_fast > 10) or (delta_post > 20)

        # comprobar que las tres medidas estén en rango de PREDIABETES (A1C debe estar presente)
        all_three_pred = (
            cf_new == "Prediabetes" and
            cp_new == "Prediabetes" and
            (new_entry['a1c'] is not None and ca_new == "Prediabetes")
        )

        if increases_significant and all_three_pred:
            mensajes.append("Empeoramiento respecto a lecturas previas: aumento significativo con todas las medidas en rango de PREDIABETES. Considerar evaluación clínica.")
            if severidad != "Crítica":
                severidad = "Advertencia"

        # indicios de prediabetes (promedios o valores actuales)
        pred_fast = (100 < new_entry['fast'] <= 125) or (100 < avg_prev_fast <= 125)
        pred_post = (141 <= new_entry['post'] <= 199) or (141 <= avg_prev_post <= 199)
        pred_a1c = (new_entry['a1c'] is not None and 5.6 <= new_entry['a1c'] <= 6.4) or (avg_prev_a1c is not None and 5.6 <= avg_prev_a1c <= 6.4)
        if pred_fast or pred_post or pred_a1c:
            mensajes.append("Indicios de PREDIABETES (orientativo). Recomendable seguimiento por profesional de la salud.")
            if severidad == "Normal":
                severidad = "Advertencia"

        # si no hay mensajes pero la clasificación indica límite alto
        if not mensajes:
            if cf_new in ("Prediabetes", "Diabetes") or cp_new in ("Límite alto", "Prediabetes", "Diabetes") or ca_new in ("Prediabetes", "Diabetes"):
                mensajes.append("La lectura actual requiere seguimiento y evaluación por el profesional de la salud.")
                severidad = "Seguimiento"

        resumen = (
            f"Lectura día {new_entry['day']} - Ayuno: {new_entry['fast']:.1f} mg/dL ({cf_new}), "
            f"2h pos: {new_entry['post']:.1f} mg/dL ({cp_new}), "
            f"A1C: {str(new_entry['a1c']) if new_entry['a1c'] is not None else 'no disponible'} ({ca_new}).\n\n"
            f"Promedios previos: Ayuno {avg_prev_fast:.1f} mg/dL, 2h pos {avg_prev_post:.1f} mg/dL"
        )
        if avg_prev_a1c is not None:
            resumen += f", A1C {avg_prev_a1c:.2f}%"
        resumen += "\n\n" + "\n".join(mensajes) if mensajes else "\n\nSin hallazgos significativos en relación con lecturas previas."
        resumen += "\n\nNota: Orientativo. Diagnóstico por profesional de la salud."

        return severidad, resumen

    def assess_with_history(self, new_entry):
        # ahora utiliza el formateador y muestra diálogo apropiado (método existente adaptado)
        severity, resumen = self.format_assessment_for_professional(new_entry)
        title_map = {
            "Crítica": "Alerta clínica: Valores críticos",
            "Advertencia": "Aviso clínico: Revisión recomendada",
            "Seguimiento": "Seguimiento: Evaluación recomendada",
            "Normal": "Evaluación: Sin hallazgos críticos"
        }
        if severity == "Crítica":
            messagebox.showerror(title_map[severity], "Para su profesional de la salud:\n\n" + resumen)
        elif severity in ("Advertencia", "Seguimiento"):
            messagebox.showwarning(title_map[severity], "Para su profesional de la salud:\n\n" + resumen)
        else:
            # actualizar panel resumen (no intrusivo)
            self.lbl_summary.config(text=resumen.replace("Para su profesional de la salud:\n\n", ""))

    def show_history_window(self):
        """
        Ventana emergente con historial completo y detalle de evaluación por fila.
        """
        if not self.readings:
            messagebox.showinfo("Historial", "No hay lecturas para mostrar.")
            return
        win = tk.Toplevel(self.root)
        win.title("Historial detallado - Para profesional")
        win.geometry("900x520")

        cols = ("día", "fecha", "ayuno", "2h pos", "A1C", "clasif ayuno", "clasif 2h", "clasif A1C")
        tree_h = ttk.Treeview(win, columns=cols, show="headings", height=14)
        for c in cols:
            tree_h.heading(c, text=c)
            tree_h.column(c, width=100, anchor="center")
        tree_h.pack(fill="both", expand=False, padx=8, pady=6)

        for r in self.readings:
            a1c_display = f"{r['a1c']:.2f}" if r['a1c'] is not None else "-"
            cf = self.classify_fast(r['fast'])
            cp = self.classify_post(r['post'])
            ca = self.classify_a1c(r['a1c']) if r['a1c'] is not None else "-"
            tree_h.insert("", "end", values=(r['day'], r['date'], f"{r['fast']:.1f}", f"{r['post']:.1f}", a1c_display, cf, cp, ca))

        txt = tk.Text(win, height=10, wrap="word", font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True, padx=8, pady=6)

        def on_sel(event):
            sel = tree_h.selection()
            if not sel:
                return
            vals = tree_h.item(sel[0], "values")
            day = int(vals[0])
            entry = next((x for x in self.readings if x['day'] == day), None)
            if entry:
                sev, resumen = self.format_assessment_for_professional(entry)
                txt.delete("1.0", tk.END)
                txt.insert(tk.END, f"Severidad: {sev}\n\n" + resumen)

        tree_h.bind("<<TreeviewSelect>>", on_sel)

    def show_selected_assessment(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Evaluación", "Seleccione una lectura en la tabla principal.")
            return
        vals = self.tree.item(sel[0], "values")
        day = int(vals[0])
        entry = next((r for r in self.readings if r['day'] == day), None)
        if not entry:
            messagebox.showerror("Error", "No se encontró la lectura seleccionada.")
            return
        severity, resumen = self.format_assessment_for_professional(entry)
        title_map = {
            "Crítica": "Alerta clínica: Valores críticos",
            "Advertencia": "Aviso clínico: Revisión recomendada",
            "Seguimiento": "Seguimiento: Evaluación recomendada",
            "Normal": "Evaluación: Sin hallazgos críticos"
        }
        if severity == "Crítica":
            messagebox.showerror(title_map[severity], "Para su profesional de la salud:\n\n" + resumen)
        else:
            messagebox.showinfo(title_map[severity], "Para su profesional de la salud:\n\n" + resumen)

    def _set_next_day_and_prefill_inputs(self):
        """
        Calcula el siguiente día disponible (1-7) según lecturas cargadas y
        prefill de campos: day_var y a1c_var (usar default_a1c si existe).
        Limpia fast/post para captura de nueva lectura.
        """
        existing_days = sorted({r['day'] for r in self.readings})
        # buscar primer día faltante entre 1 y 90
        next_day = None
        for d in range(1, 91):
            if d not in existing_days:
                next_day = d
                break
        if next_day is None:
            # si ya estaban todos, poner el siguiente lógico (hasta 90)
            next_day = min(90, (existing_days[-1] if existing_days else 0) + 1)
        self.next_day = next_day
        try:
            self.day_var.set(self.next_day)
        except:
            pass
        # poner A1C por defecto si existe
        if self.default_a1c is not None:
            try:
                self.a1c_var.set(f"{self.default_a1c:.2f}")
            except:
                self.a1c_var.set(str(self.default_a1c))
        else:
            self.a1c_var.set("")
        # limpiar entradas de glucosa para nueva captura
        self.fast_var.set("")
        self.post_var.set("")

    def _create_evaluation_window(self):
        eval_win = tk.Toplevel(self.root)
        eval_win.title("🏥 Evaluación IA Avanzada - 15 Factores")
        # ventana inicial compacta para evitar espacio desperdiciado
        eval_win.geometry("660x460")
        eval_win.minsize(620, 420)
        eval_win.resizable(True, True)
        eval_win.config(bg="#f5f5f5")
        
        # Hacer la ventana modal
        eval_win.transient(self.root)
        eval_win.grab_set()
        
        # ==== HEADER MEJORADO ====
        header_frame = tk.Frame(eval_win, bg="#1e3a8a", height=50)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🏥 Evaluación Integral de Riesgo de Diabetes",
            font=("Segoe UI", 16, "bold"),
            bg="#1e3a8a",
            fg="white"
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Análisis basado en IA de 15 factores clínicos",
            font=("Segoe UI", 10),
            bg="#1e3a8a",
            fg="#e0e7ff"
        )
        subtitle_label.pack(pady=(0, 10))
        
        # ==== CONTENEDOR PRINCIPAL ====
        main_frame = tk.Frame(eval_win, bg="#f5f5f5")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Etiqueta de instrucciones
        instruction_label = tk.Label(
            main_frame,
            text="📋 Responde las siguientes preguntas:",
            font=("Segoe UI", 10),
            bg="#f5f5f5",
            fg="#1f2937",
            justify="left"
        )
        instruction_label.pack(anchor="w", pady=(0, 8))
        
        # ==== ÁREA DE PREGUNTAS CON DISEÑO MEJORADO ====
        canvas_frame = tk.Frame(main_frame, bg="#f9fafb")
        # el marco ocupará todo el ancho, pero el canvas dentro tendrá ancho fijo
        canvas_frame.pack(fill="both", expand=True)
        canvas_frame.config(padx=4, pady=4)
        
        # el argumento `bg_color` no es válido para Canvas; usar solo `bg`
        canvas = tk.Canvas(
            canvas_frame,
            bg="#ffffff",
            relief="solid",
            bd=1,
            highlightthickness=0,
        )
        # ancho fijo para que no se estire demasiado
        canvas.config(width=620, bg="#ffffff", highlightthickness=1, highlightbackground="#d1d5db")
        
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        # el marco de preguntas se limita a un ancho fijo y usa gris suave de fondo
        scrollable_frame = tk.Frame(canvas, bg="#f9fafb", width=580)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # fijar ancho de ventana interna para centrar contenido y limitar expansión
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=600)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # no permitir que el canvas se estire horizontalmente
        canvas.pack(side="left", fill="y", padx=(10,0), pady=5)
        scrollbar.pack(side="right", fill="y", pady=5)
        
        # Almacenar referencias
        eval_win.questions_frame = scrollable_frame
        eval_win.questions_data = []
        eval_win.question_count = tk.StringVar(value="Preguntas respondidas: 0")
        
        # ==== FOOTER CON CONTADOR ====
        footer_frame = tk.Frame(eval_win, bg="#f3f4f6", height=30)
        footer_frame.pack(fill="x", padx=15, pady=(10, 0))
        footer_frame.pack_propagate(False)
        
        counter_label = tk.Label(
            footer_frame,
            textvariable=eval_win.question_count,
            font=("Segoe UI", 9),
            bg="#f3f4f6",
            fg="#6b7280"
        )
        counter_label.pack(anchor="w", padx=10, pady=10)
        
        return eval_win

    def _add_question_to_window(self, eval_win, question_text, answer_value, style="normal"):
        """Agrega una pregunta con su respuesta al panel visible - Diseño mejorado."""
        # Actualizar contador
        current_count = len(eval_win.questions_data)
        eval_win.question_count.set(f"✓ Preguntas respondidas: {current_count + 1}")
        eval_win.questions_data.append((question_text, answer_value))
        
        # Marco para cada pregunta-respuesta
        # cada tarjeta se limita al ancho del panel
        q_frame = tk.Frame(eval_win.questions_frame, bg="#ffffff", width=540)
        # más margen horizontal para que quede centrado en el fondo gris
        q_frame.pack(fill="x", pady=4, padx=20)
        q_frame.pack_propagate(False)
        
        # Número de pregunta con icono
        # marco del número sin color llamativo
        num_frame = tk.Frame(q_frame, bg="#ffffff")
        num_frame.pack(fill="x", pady=(0, 4))
        
        num_label = tk.Label(
            num_frame,
            text=f"#{len(eval_win.questions_data)} ❓",
            font=("Segoe UI", 8, "bold"),
            bg="#ffffff",
            fg="#1e40af"
        )
        num_label.pack(anchor="w", padx=10, pady=6)
        
        # Pregunta con mejor formato
        question_label = tk.Label(
            q_frame,
            text=question_text,
            font=("Segoe UI", 10, "bold"),
            bg="#ffffff",
            fg="#1f2937",
            justify="left",
            wraplength=600
        )
        question_label.pack(anchor="w", padx=10, pady=(0, 6))
        
        # Respuesta con estilo según tipo
        if style == "normal":
            response_bg = "#ecfdf5"
            response_color = "#065f46"
            response_icon = "✅"
        elif style == "warning":
            response_bg = "#fef2f2"
            response_color = "#7f1d1d"
            response_icon = "⚠️"
        else:
            response_bg = "#eff6ff"
            response_color = "#0c4a6e"
            response_icon = "ℹ️"
        
        response_frame = tk.Frame(q_frame, bg=response_bg, bd=1, relief="solid")
        response_frame.pack(fill="x", padx=8, pady=(0, 8))
        
        response_label = tk.Label(
            response_frame,
            text=f"{response_icon} {answer_value}",
            font=("Segoe UI", 10),
            bg=response_bg,
            fg=response_color,
            justify="left",
            wraplength=600
        )
        response_label.pack(anchor="w", padx=10, pady=6)
        
        # Separador visual mejorado
        separator = tk.Frame(q_frame, bg="#e5e7eb", height=1)
        separator.pack(fill="x", pady=(4, 0))

    def _ask_integer_in_window(self, eval_win, prompt, min_val=8, max_val=120):
        """Pide un valor entero manteniendo visible la pregunta."""
        value = simpledialog.askinteger("Información Personal", prompt, minvalue=min_val, maxvalue=max_val, parent=eval_win)
        return value

    def _ask_float_in_window(self, eval_win, prompt, title):
        """Pide un valor decimal manteniendo visible la pregunta."""
        while True:
            value_str = simpledialog.askstring(title, prompt, parent=eval_win)
            if value_str is None:
                return None
            try:
                value = float(value_str)
                if value < 0 or value > 500:
                    messagebox.showwarning("Valor inválido", "Por favor ingrese un valor válido.", parent=eval_win)
                    continue
                return value
            except ValueError:
                messagebox.showwarning("Error", "Ingrese un número válido.", parent=eval_win)

    def evaluate_diabetes_risk(self):
        """
        Evaluación integral avanzada de riesgo de diabetes con 15 características.
        Incluye prediabetes infantil, factores epidemiológicos y recomendaciones precisas.
        Usa una ventana personalizada para mantener visibles preguntas y respuestas.
        """
        if not self.readings:
            messagebox.showinfo("Evaluación IA Avanzada", "No hay lecturas de glucosa disponibles para evaluar.")
            return

        # Crear ventana de evaluación
        eval_win = self._create_evaluation_window()

        # PASO 1: Obtener edad (de BD o preguntar)
        age = None
        if self.current_participant_id is not None:
            try:
                self.cursor.execute("SELECT edad FROM participantes WHERE id = ?", (self.current_participant_id,))
                row = self.cursor.fetchone()
                if row and row[0] is not None:
                    age = int(row[0])
            except Exception:
                age = None
        
        if age is None:
            age = self._ask_integer_in_window(
                eval_win,
                "¿Cuál es la edad del participante? (años):\n\n"
                "Nota: Para niños (4-14 años), se evalúa Tipo 1\n"
                "Para jóvenes (20-44), se evalúa Tipo 2 temprana\n"
                "Para mayores (45+), se evalúa Tipo 2 clásica",
                min_val=4,
                max_val=120
            )
            if age is None:
                eval_win.destroy()
                return
        
        self._add_question_to_window(eval_win, "¿Cuál es la edad?", f"{age} años", "normal")
        
        # PASO 2: Calcular promedios de glucosa
        fast_vals = [r['fast'] for r in self.readings]
        post_vals = [r['post'] for r in self.readings]
        a1c_vals = [r['a1c'] for r in self.readings if r['a1c'] is not None]
        avg_fast = statistics.mean(fast_vals)
        avg_post = statistics.mean(post_vals)
        avg_a1c = statistics.mean(a1c_vals) if a1c_vals else 5.0
        
        self._add_question_to_window(eval_win, "Glucosa promedio en ayunas", f"{avg_fast:.1f} mg/dL", "normal")
        self._add_question_to_window(eval_win, "Glucosa promedio post-prandial", f"{avg_post:.1f} mg/dL", "normal")
        self._add_question_to_window(eval_win, "A1C promedio", f"{avg_a1c:.1f}%", "normal")
        
        # PASO 3: Evaluar colesterol completo
        messagebox.showinfo(
            "Datos de Colesterol - Evaluación Avanzada",
            "A continuación, ingrese sus valores de colesterol en mg/dL.\n\n"
            "Niveles óptimos de referencia:\n"
            "• LDL (malo): < 100 mg/dL (óptimo), < 70 mg/dL (muy protector)\n"
            "• HDL (bueno): ≥ 60 mg/dL (protector fuerte)\n"
            "• Triglicéridos: < 150 mg/dL\n"
            "• Colesterol Total: < 200 mg/dL",
            parent=eval_win
        )
        
        # Solicitar datos de colesterol con validación mejorada
        ldl = self._ask_float_in_window(
            eval_win,
            "Ingrese el valor de LDL en mg/dL\n(ej: 120)\n\nÓptimo: < 100 mg/dL\nMuy protector: < 70 mg/dL",
            "Colesterol LDL (Malo)"
        )
        if ldl is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuál es su LDL (Malo)?", f"{ldl:.1f} mg/dL", "normal")
        
        hdl = self._ask_float_in_window(
            eval_win,
            "Ingrese el valor de HDL en mg/dL\n(ej: 50)\n\nProtector fuerte: ≥ 60 mg/dL\nProtector: ≥ 50 mg/dL",
            "Colesterol HDL (Bueno)"
        )
        if hdl is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuál es su HDL (Bueno)?", f"{hdl:.1f} mg/dL", "normal")
        
        triglycerides = self._ask_float_in_window(
            eval_win,
            "Ingrese el valor de triglicéridos en mg/dL\n(ej: 140)\n\nÓptimo: < 150 mg/dL\nElevado: 150-199 mg/dL",
            "Triglicéridos"
        )
        if triglycerides is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuál es su Triglicéridos?", f"{triglycerides:.1f} mg/dL", "normal")
        
        total_chol = ldl + hdl + (triglycerides / 5.0)
        self._add_question_to_window(eval_win, "Colesterol Total (calculado)", f"{total_chol:.1f} mg/dL", "normal")
        
        # PASO 4: IMC y presión arterial
        bmi = self._ask_float_in_window(
            eval_win,
            "Ingrese su IMC (kg/m²)\n(ej: 25.5)\n\nO calcule: peso(kg) / [altura(m)]²\n\nNormal: 18.5-24.9\nSobrepeso: 25-29.9\nObesidad: ≥ 30",
            "Índice de Masa Corporal (IMC)"
        )
        if bmi is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuál es su IMC?", f"{bmi:.1f} kg/m²", "normal")
        
        systolic = self._ask_float_in_window(
            eval_win,
            "Ingrese la presión sistólica (mmHg)\n(ej: 120)\n\nNormal: < 120 mmHg\nElevada: 120-129 mmHg\nHipertensión: ≥ 130 mmHg",
            "Presión Arterial Sistólica"
        )
        if systolic is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuál es su presión sistólica?", f"{systolic:.0f} mmHg", "normal")
        
        diastolic = self._ask_float_in_window(
            eval_win,
            "Ingrese la presión diastólica (mmHg)\n(ej: 80)\n\nNormal: < 80 mmHg\nElevada: 80-89 mmHg\nHipertensión: ≥ 90 mmHg",
            "Presión Arterial Diastólica"
        )
        if diastolic is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuál es su presión diastólica?", f"{diastolic:.0f} mmHg", "normal")
        
        # PASO 5: Antecedentes familiares y estilo de vida
        has_diabetes_family = messagebox.askyesno(
            "Antecedentes Familiares",
            "¿Tiene antecedentes de diabetes en familiares de primer grado?\n"
            "(padres, hermanos, hijos)\n\n"
            "Esto aumenta significativamente el riesgo.",
            parent=eval_win
        )
        family_history = 1 if has_diabetes_family else 0
        family_risk_factor = 1.0
        family_details = {
            'has_family_history': has_diabetes_family,
            'family_count': 0,
            'relationship_types': [],
            'average_onset_age': None,
            'risk_factor': 1.0
        }
        self._add_question_to_window(eval_win, "¿Antecedentes familiares de diabetes?", f"{'Sí' if has_diabetes_family else 'No'}", "normal")

        if has_diabetes_family:
            family_count = self._ask_integer_in_window(
                eval_win,
                "¿Cuántos familiares de primer grado tienen diabetes?\n"
                "Ejemplos: padres, hermanos, hijos",
                min_val=1,
                max_val=10
            )
            if family_count is None:
                family_count = 1
            family_details['family_count'] = family_count
            self._add_question_to_window(eval_win, "¿Cuántos familiares con diabetes?", f"{family_count}", "warning")

            relationships = []
            if messagebox.askyesno("Parentesco", "¿Tiene padre(s) con diabetes?", parent=eval_win):
                relationships.append("padre")
            if messagebox.askyesno("Parentesco", "¿Tiene madre con diabetes?", parent=eval_win):
                relationships.append("madre")
            if messagebox.askyesno("Parentesco", "¿Tiene hermano(s) con diabetes?", parent=eval_win):
                relationships.append("hermano")
            if messagebox.askyesno("Parentesco", "¿Tiene hijo(s) con diabetes?", parent=eval_win):
                relationships.append("hijo")

            family_details['relationship_types'] = relationships
            if relationships:
                self._add_question_to_window(eval_win, "Tipo de parentesco", f"{', '.join(relationships)}", "warning")

            onset_ages = []
            for rel in relationships:
                age_onset = self._ask_integer_in_window(
                    eval_win,
                    f"¿A qué edad aproximadamente desarrolló diabetes su {rel}?\n"
                    "(Si no sabe, estime lo mejor posible)",
                    min_val=10,
                    max_val=90
                )
                if age_onset is not None:
                    onset_ages.append(age_onset)

            if onset_ages:
                family_details['average_onset_age'] = sum(onset_ages) / len(onset_ages)
                self._add_question_to_window(eval_win, "Edad promedio de aparición", f"{family_details['average_onset_age']:.0f} años", "warning")

            base_risk = 1.0
            if family_count >= 2:
                base_risk += 0.5
            if "padre" in relationships or "madre" in relationships:
                base_risk += 0.3
            if "hermano" in relationships:
                base_risk += 0.2
            if onset_ages and sum(onset_ages)/len(onset_ages) < 50:
                base_risk += 0.2

            family_risk_factor = min(3.0, max(1.0, base_risk))
            family_details['risk_factor'] = family_risk_factor
        
        exercise_weekly = self._ask_float_in_window(
            eval_win,
            "¿Cuántos minutos de ejercicio moderado realiza por semana?\n"
            "Recomendación: mínimo 150 minutos\n(ej: 180, 300)\n\n"
            "Incluye: caminar rápido, ciclismo, natación, etc.",
            "Actividad Física Semanal"
        )
        if exercise_weekly is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Cuántos minutos de ejercicio semanales?", f"{exercise_weekly:.0f} minutos", "normal")
        
        diet_quality = self._ask_integer_in_window(
            eval_win,
            "Califique su adherencia a una dieta saludable:\n"
            "1 = Pobre | 2 = Regular | 3 = Buena | 4 = Muy Buena | 5 = Excelente",
            min_val=1,
            max_val=5
        )
        if diet_quality is None:
            eval_win.destroy()
            return
        self._add_question_to_window(eval_win, "¿Calidad de su dieta?", f"{diet_quality}/5", "normal")
        
        # PASO 6: Calcular care_score avanzado
        care_score = 2.0
        care_score -= min(1.2, exercise_weekly / 200.0)
        care_score -= (diet_quality - 1) * 0.25
        care_score += family_history * 0.3
        care_score = max(0.0, min(2.0, care_score))
        
        # PASO 7: Preparar características para el modelo avanzado (15 features)
        features = [
            age, avg_fast, avg_post, avg_a1c, bmi,
            ldl, hdl, triglycerides, total_chol,
            systolic, diastolic, care_score,
            family_history, exercise_weekly, diet_quality
        ]
        
        # PASO 8: Evaluación con modelo avanzado
        model = self.get_diabetes_model()
        if model is None:
            messagebox.showerror("Error IA", "No se pudo inicializar el modelo IA avanzado.", parent=eval_win)
            eval_win.destroy()
            return

        prob, stage, risk_factors, recommendations = assess_diabetes_risk_comprehensive(features, model)
        timeframe = map_probability_to_timeframe(prob, age=age, avg_glucose=avg_fast, ldl=ldl, hdl=hdl, stage=stage, family_risk_factor=family_risk_factor)
        symptoms = possible_symptoms_by_probability(prob, stage=stage, age=age)
        lipid_risk, lipid_details, lipid_risk_score = classify_lipid_profile(ldl, hdl, triglycerides, total_chol)
        lipid_recommendations = generate_lipid_recommendations(ldl, hdl, triglycerides, risk_score=lipid_risk_score)
        
        # PASO 12: Generar reporte integral avanzado
        stage_descriptions = {
            0: "Normal - Sin evidencia de diabetes",
            1: "Prediabetes - Riesgo elevado de diabetes",
            2: "Diabetes - Requiere manejo médico activo"
        }
        
        report = (
            "╔════════════════════════════════════════════════════════════════╗\n"
            "║        EVALUACIÓN AVANZADA DE RIESGO DE DIABETES (15 FACTORES) ║\n"
            "╚════════════════════════════════════════════════════════════════╝\n\n"
            f"📊 RESULTADO DE EVALUACIÓN\n"
            f"   Probabilidad de Diabetes: {prob*100:.1f}%\n"
            f"   Etapa Actual: {stage_descriptions[stage]}\n"
            f"   Estimación Temporal: {timeframe}\n\n"
            
            f"❤️  PERFIL CARDIOVASCULAR AVANZADO\n"
            f"   Riesgo Cardiovascular: {lipid_risk} (Score: {lipid_risk_score:.1f}/10)\n"
        )
        
        for detail in lipid_details:
            report += f"   {detail}\n"
        
        report += f"\n📋 FACTORES DE RIESGO IDENTIFICADOS\n"
        if risk_factors:
            for factor in risk_factors:
                report += f"   ⚠️  {factor}\n"
        else:
            report += f"   ✅ No se identificaron factores de riesgo mayores\n"
        
        report += f"\n🩺 SÍNTOMAS POSIBLES\n"
        for sym in symptoms:
            report += f"   {sym}\n"
        
        # RESUMEN DE DATOS INGRESADOS
        report += f"\n📊 RESUMEN DE DATOS EVALUADOS\n"
        report += f"   • Edad: {age} años\n"
        report += f"   • Glucosa en ayunas promedio: {avg_fast:.1f} mg/dL\n"
        report += f"   • Glucosa post-prandial promedio: {avg_post:.1f} mg/dL\n"
        report += f"   • A1C promedio: {avg_a1c:.1f}%\n"
        report += f"   • IMC: {bmi:.1f} kg/m²\n"
        report += f"   • Presión arterial: {systolic:.0f}/{diastolic:.0f} mmHg\n"
        report += f"   • Colesterol LDL: {ldl:.1f} mg/dL\n"
        report += f"   • Colesterol HDL: {hdl:.1f} mg/dL\n"
        report += f"   • Triglicéridos: {triglycerides:.1f} mg/dL\n"
        report += f"   • Colesterol Total: {total_chol:.1f} mg/dL\n"
        report += f"   • Ejercicio semanal: {exercise_weekly:.0f} minutos\n"
        report += f"   • Calidad de dieta: {diet_quality}/5\n"
        if family_details['has_family_history']:
            report += f"   • Antecedentes familiares: Sí ({family_details['family_count']} familiar(es))\n"
            if family_details['relationship_types']:
                report += f"   • Parentesco: {', '.join(family_details['relationship_types'])}\n"
            if family_details['average_onset_age']:
                report += f"   • Edad promedio de aparición: {family_details['average_onset_age']:.0f} años\n"
            report += f"   • Factor de riesgo familiar: {family_details['risk_factor']:.1f}x\n"
        else:
            report += f"   • Antecedentes familiares: No\n"
        
        report += f"\n💊 RECOMENDACIONES ESPECÍFICAS\n"
        for rec in recommendations:
            report += f"   {rec}\n"
        
        report += f"\n💉 RECOMENDACIONES POR COLESTEROL\n"
        for rec in lipid_recommendations:
            report += f"   {rec}\n"
        
        report += (
            f"\n📌 RECOMENDACIONES GENERALES\n"
            f"   • Realizar chequeos médicos regulares según riesgo\n"
            f"   • Seguir las orientaciones del profesional de salud\n"
            f"   • Mantener registro continuo de glucosa y parámetros\n"
            f"   • Aumentar actividad física y mejorar hábitos alimenticios\n\n"
            f"⚠️  NOTA: Esta evaluación usa IA avanzada con 15 factores de riesgo.\n"
            f"     El diagnóstico definitivo requiere evaluación médica profesional."
        )
        
        # PASO 13: Mostrar reporte en ventana con scroll (mejor visualización)
        self._show_report_window(report)
        eval_win.destroy()

    def _show_report_window(self, report_text):
        """Muestra el reporte completo en una ventana con scroll para mejor visualización - Diseño mejorado."""
        report_win = tk.Toplevel(self.root)
        report_win.title("📊 Reporte Completo - Evaluación IA Avanzada")
        report_win.geometry("1100x750")
        report_win.resizable(True, True)
        report_win.config(bg="#f5f5f5")
        
        # ==== HEADER MEJORADO ====
        header_frame = tk.Frame(report_win, bg="#10b981", height=80)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="📊 Reporte Completo de Evaluación",
            font=("Segoe UI", 16, "bold"),
            bg="#10b981",
            fg="white"
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            header_frame,
            text="Análisis integral de 15 factores clínicos - Generado por IA Avanzada",
            font=("Segoe UI", 10),
            bg="#10b981",
            fg="#d1fae5"
        )
        subtitle_label.pack(pady=(0, 10))
        
        # ==== BARRA DE HERRAMIENTAS ====
        toolbar_frame = tk.Frame(report_win, bg="#ffffff", height=60)
        toolbar_frame.pack(fill="x", padx=0, pady=0)
        toolbar_frame.pack_propagate(False)
        
        # Botones mejorados
        button_frame = tk.Frame(toolbar_frame, bg="#ffffff")
        button_frame.pack(anchor="w", padx=15, pady=12)
        
        # Botón Copiar
        copy_btn = tk.Button(
            button_frame,
            text="📋 Copiar al Portapapeles",
            command=lambda: self._copy_to_clipboard(report_text),
            bg="#3b82f6",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8,
            relief="flat",
            cursor="hand2",
            activebackground="#2563eb",
            activeforeground="white"
        )
        copy_btn.pack(side="left", padx=8)
        
        # Botón Guardar
        save_btn = tk.Button(
            button_frame,
            text="💾 Guardar como Archivo",
            command=lambda: self._save_report(report_text),
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8,
            relief="flat",
            cursor="hand2",
            activebackground="#059669",
            activeforeground="white"
        )
        save_btn.pack(side="left", padx=8)
        
        # Botón Cerrar
        close_btn = tk.Button(
            button_frame,
            text="✖ Cerrar",
            command=report_win.destroy,
            bg="#ef4444",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8,
            relief="flat",
            cursor="hand2",
            activebackground="#dc2626",
            activeforeground="white"
        )
        close_btn.pack(side="right", padx=8)
        
        # ==== ÁREA DE CONTENIDO ====
        content_frame = tk.Frame(report_win, bg="#f5f5f5")
        content_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Área de texto con scroll
        text_frame = tk.Frame(content_frame, bg="#ffffff", relief="solid", bd=1)
        text_frame.pack(fill="both", expand=True)
        
        text_widget = tk.Text(
            text_frame,
            wrap="word",
            font=("Courier New", 10),
            bg="#ffffff",
            fg="#1f2937",
            padx=12,
            pady=12,
            relief="flat"
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Insertar reporte con colores
        text_widget.insert("1.0", report_text)
        
        # Colorear secciones importantes
        text_widget.tag_configure("titulo", foreground="#1e40af", font=("Courier New", 10, "bold"))
        text_widget.tag_configure("resultado", foreground="#059669", font=("Courier New", 10, "bold"))
        text_widget.tag_configure("warning", foreground="#dc2626", font=("Courier New", 10, "bold"))
        
        text_widget.config(state="disabled")  # Lectura solamente
        
        # ==== STATUS BAR MEJORADO ====
        status_frame = tk.Frame(report_win, bg="#f3f4f6", height=50)
        status_frame.pack(fill="x", padx=0, pady=0)
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(
            status_frame,
            text="✅ Reporte generado exitosamente - Usa Scroll o Ctrl+A para seleccionar todo",
            font=("Segoe UI", 10),
            bg="#f3f4f6",
            fg="#4b5563"
        )
        status_label.pack(anchor="w", padx=15, pady=12)

    def _copy_to_clipboard(self, text):
        """Copia el reporte al portapapeles."""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            messagebox.showinfo("✅ Éxito", "Reporte copiado al portapapeles correctamente.")
        except Exception as e:
            messagebox.showerror("❌ Error", f"No se pudo copiar: {e}")

    def _save_report(self, text):
        """Guarda el reporte en un archivo."""
        try:
            from datetime import datetime
            filename = f"Reporte_Diabetes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=filename, 
                                                    filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                messagebox.showinfo("✅ Éxito", f"Reporte guardado en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("❌ Error", f"No se pudo guardar: {e}")


    def train_model_from_db(self):
        """Entrena el modelo usando los participantes y sus lecturas en la BD.

        Etiquetas (heurística):
        - positivo (1) si A1C promedio >= 6.5 o glucosa ayuno promedio >125 o 2h pos promedio >199
        - negativo (0) en caso contrario
        """
        # construir dataset
        X = []
        y = []
        try:
            self.cursor.execute("SELECT id, edad FROM participantes")
            parts = self.cursor.fetchall()
        except Exception as e:
            messagebox.showerror("Entrenar IA", f"Error leyendo participantes: {e}")
            return

        for pid, edad in parts:
            self.cursor.execute("SELECT ayuno, pos2h, a1c FROM lecturas WHERE participante_id = ?", (pid,))
            rows = self.cursor.fetchall()
            if not rows:
                continue
            fast_vals = [r[0] for r in rows if r[0] is not None]
            post_vals = [r[1] for r in rows if r[1] is not None]
            a1c_vals = [r[2] for r in rows if r[2] is not None]
            if not fast_vals and not post_vals and not a1c_vals:
                continue
            avg_fast = statistics.mean(fast_vals) if fast_vals else 0.0
            avg_post = statistics.mean(post_vals) if post_vals else 0.0
            avg_a1c = statistics.mean(a1c_vals) if a1c_vals else 5.0

            # edad: si nula, saltar
            if edad is None:
                continue
            try:
                age_val = int(edad)
            except:
                continue

            # care_score: heurística simple basada en ausencia de datos (tratada como intermedia)
            # aquí usamos 1.0 por defecto; si quieres, podríamos inferirlo de campos adicionales
            care_score = 1.0

            # etiqueta por umbrales clínicos
            label = 1 if (avg_a1c >= 6.5 or avg_fast > 125 or avg_post > 199) else 0

            X.append([age_val, avg_fast, avg_post, avg_a1c, care_score])
            y.append(label)

        if len(X) < 8:
            messagebox.showinfo("Entrenar IA", "No hay suficientes datos con edad y lecturas para entrenar (mínimo 8 participantes con lecturas).")
            return

        # entrenar
        try:
            model = LogisticRegressionModel(n_features=5)
            model.fit(X, y, lr=0.0008, epochs=4000)
            self.diabetes_model = model
            # guardar en disco
            with open("diabetes_model.pkl", "wb") as f:
                pickle.dump(model.w, f)
            messagebox.showinfo("Entrenar IA", f"Modelo entrenado con {len(X)} participantes. Pesos guardados en 'diabetes_model.pkl'.")
        except Exception as e:
            messagebox.showerror("Entrenar IA", f"Error durante el entrenamiento: {e}")

    def show_app_options(self):
        """Muestra una ventana con todas las opciones y botones disponibles en la aplicación."""
        options = [
            "Registrar participante (panel izquierdo)",
            "Seleccionar participante (combobox)",
            "Agregar lectura diaria: Día, Glucosa ayuno, 2h pos, A1C (opcional)",
            "Resetear semana (borra lecturas del participante seleccionado)",
            "Exportar CSV (exporta lecturas visibles)",
            "Historial detallado (ventana emergente)",
            "Resumen semanal completo (botón 'Resumen semanal completo')",
            "Evaluar lectura seleccionada (botón 'Evaluar lectura seleccionada')",
            "Evaluar riesgo IA Avanzada (botón 'Evaluar riesgo IA Avanzada') - modelo con 15 factores de riesgo",
            "Entrenar IA con BD (botón 'Entrenar IA con BD') - entrena con datos en la base de datos y guarda pesos",
            "Ver opciones (este panel)",
        ]

        win = tk.Toplevel(self.root)
        win.title("Opciones disponibles")
        win.geometry("560x360")

        frm = ttk.Frame(win, padding=10)
        frm.pack(fill="both", expand=True)

        lbl = ttk.Label(frm, text="Opciones y botones disponibles en la aplicación:", style="Header.TLabel")
        lbl.pack(anchor="w", pady=(0,8))

        txt = tk.Text(frm, wrap="word", height=15, font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True)
        for i, line in enumerate(options, start=1):
            txt.insert(tk.END, f"{i}. {line}\n")
        txt.configure(state="disabled")

        btn_close = ttk.Button(frm, text="Cerrar", command=win.destroy)
        btn_close.pack(pady=8)

def main():
    root = tk.Tk()
    app = GlucoseMonitorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()