import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import statistics
import csv
from datetime import datetime
import sqlite3
from diabetes_model import (
    train_synthetic_model,
    map_probability_to_timeframe,
    possible_symptoms_by_probability,
    LogisticRegressionModel,
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

        # modelo IA (entrenado con dataset sintético al iniciarse la app)
        try:
            self.diabetes_model = train_synthetic_model()
        except Exception:
            self.diabetes_model = None

        self._build_ui()
    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=28)
        style.configure("Primary.TButton", foreground="white", background="#2E8B57", font=("Segoe UI", 11, "bold"))
        style.map("Primary.TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#246b44')])
        style.configure("Danger.TButton", foreground="white", background="#C0392B", font=("Segoe UI", 11, "bold"))
        style.map("Danger.TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#a8322a')])

        self.root.geometry("1000x660")
        self.root.minsize(920, 600)
        try:
            self.root.configure(bg="#f4f7f6")
        except:
            pass

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=12, pady=12)

        # Contenedor izquierdo scrollable: usamos un Canvas + scrollbar
        left_container = ttk.Frame(paned, width=360)
        paned.add(left_container, weight=0)

        canvas_left = tk.Canvas(left_container, highlightthickness=0)
        vscroll = ttk.Scrollbar(left_container, orient="vertical", command=canvas_left.yview)
        canvas_left.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas_left.pack(side="left", fill="both", expand=True)

        # frame real donde añadiremos widgets (mantendremos nombre 'left')
        left = ttk.Frame(canvas_left, width=360)
        canvas_left.create_window((0, 0), window=left, anchor="nw")

        def _left_configure(event):
            canvas_left.configure(scrollregion=canvas_left.bbox("all"))

        left.bind("<Configure>", _left_configure)

        # Nuevo: panel de registro/selección de participante
        frm_part = ttk.LabelFrame(left, text="Participante / Proyecto", padding=10)
        frm_part.pack(fill="x", padx=8, pady=8)

        ttk.Label(frm_part, text="Seleccionar participante:").grid(row=0, column=0, sticky="w")
        self.participante_var = tk.StringVar()
        self.participante_cb = ttk.Combobox(frm_part, textvariable=self.participante_var, state="readonly", width=28)
        self.participante_cb.grid(row=0, column=1, padx=6, pady=4, sticky="w")
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

        frm_inputs = ttk.LabelFrame(left, text="Agregar lectura diaria", padding=14)
        frm_inputs.pack(fill="x", padx=8, pady=8)

        ttk.Label(frm_inputs, text="Día (1-90):", style="TLabel").grid(row=0, column=0, sticky="w", pady=4)
        self.day_var = tk.IntVar(value=self.next_day)
        self.ent_day = ttk.Entry(frm_inputs, textvariable=self.day_var, width=8, font=("Segoe UI", 11))
        self.ent_day.grid(row=0, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(frm_inputs, text="Glucosa en ayunas (mg/dL):").grid(row=1, column=0, sticky="w", pady=4)
        self.fast_var = tk.StringVar()
        self.ent_fast = ttk.Entry(frm_inputs, textvariable=self.fast_var, width=12, font=("Segoe UI", 11))
        self.ent_fast.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(frm_inputs, text="Glucosa 2h pos comida (mg/dL):").grid(row=2, column=0, sticky="w", pady=4)
        self.post_var = tk.StringVar()
        self.ent_post = ttk.Entry(frm_inputs, textvariable=self.post_var, width=12, font=("Segoe UI", 11))
        self.ent_post.grid(row=2, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(frm_inputs, text="A1C (% - opcional):").grid(row=3, column=0, sticky="w", pady=4)
        self.a1c_var = tk.StringVar()
        self.ent_a1c = ttk.Entry(frm_inputs, textvariable=self.a1c_var, width=12, font=("Segoe UI", 11))
        self.ent_a1c.grid(row=3, column=1, padx=8, pady=4, sticky="w")

        btn_frame = ttk.Frame(frm_inputs)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(10,0), sticky="we")
        btn_frame.columnconfigure((0,1), weight=1)

        add_border = tk.Frame(btn_frame, bg="#2E8B57", bd=0)
        add_border.grid(row=0, column=0, padx=(0,6), sticky="we")
        add_border.columnconfigure(0, weight=1)
        btn_add = ttk.Button(add_border, text="Agregar lectura", command=self.add_reading, style="Primary.TButton")
        btn_add.grid(row=0, column=0, padx=3, pady=3, sticky="we")

        reset_border = tk.Frame(btn_frame, bg="#C0392B", bd=0)
        reset_border.grid(row=0, column=1, sticky="we")
        reset_border.columnconfigure(0, weight=1)
        btn_reset = ttk.Button(reset_border, text="Resetear semana", command=self.reset_week, style="Danger.TButton")
        btn_reset.grid(row=0, column=0, padx=3, pady=3, sticky="we")

        util_frame = ttk.Frame(frm_inputs)
        util_frame.grid(row=5, column=0, columnspan=2, pady=8, sticky="we")
        util_frame.columnconfigure((0,1), weight=1)

        btn_export = ttk.Button(util_frame, text="Exportar CSV", command=self.export_csv)
        btn_export.grid(row=0, column=0, padx=(0,6), sticky="we")
        btn_history = ttk.Button(util_frame, text="Historial detallado", command=self.show_history_window)
        btn_history.grid(row=0, column=1, sticky="we")

        frm_summary = ttk.LabelFrame(left, text="Resumen rápido (para profesional)", padding=10)
        frm_summary.pack(fill="both", expand=False, padx=8, pady=8)

        self.lbl_summary = ttk.Label(frm_summary, text="No hay lecturas aún.", wraplength=320, justify="left")
        self.lbl_summary.pack(fill="both", expand=True, padx=6, pady=6)

        btn_full_summary = ttk.Button(frm_summary, text="Resumen semanal completo", command=self.show_summary)
        btn_full_summary.pack(fill="x", pady=6)

        frm_expert = ttk.LabelFrame(left, text="Herramientas para profesional", padding=8)
        frm_expert.pack(fill="both", expand=False, padx=8, pady=8)

        btn_eval_sel = ttk.Button(frm_expert, text="Evaluar lectura seleccionada", command=self.show_selected_assessment)
        btn_eval_sel.pack(fill="x", pady=4)
        btn_eval_ai = ttk.Button(frm_expert, text="Evaluar riesgo IA", command=self.evaluate_diabetes_risk)
        btn_eval_ai.pack(fill="x", pady=4)
        btn_train_db = ttk.Button(frm_expert, text="Entrenar IA con BD", command=self.train_model_from_db)
        btn_train_db.pack(fill="x", pady=4)
        btn_show_options = ttk.Button(frm_expert, text="Ver opciones", command=self.show_app_options)
        btn_show_options.pack(fill="x", pady=4)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        header = ttk.Label(right, text="Historial de la semana", style="Header.TLabel")
        header.pack(anchor="w", padx=6, pady=(6,0))

        cols = ("día", "fecha", "ayuno (mg/dL)", "2h pos (mg/dL)", "A1C (%)", "clasif ayuno", "clasif 2h", "clasif A1C")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("día", width=60, anchor="center")
        self.tree.column("fecha", width=100, anchor="center")
        self.tree.column("ayuno (mg/dL)", width=110, anchor="center")
        self.tree.column("2h pos (mg/dL)", width=120, anchor="center")
        self.tree.column("A1C (%)", width=80, anchor="center")
        self.tree.column("clasif ayuno", width=120, anchor="center")
        self.tree.column("clasif 2h", width=120, anchor="center")
        self.tree.column("clasif A1C", width=120, anchor="center")

        self.tree.pack(fill="both", expand=True, padx=8, pady=6)
        # Ahora que el tree ya existe, inicializar valores del combobox y cargar lecturas
        self._refresh_participante_combobox()

        frm_detail = ttk.LabelFrame(right, text="Detalle y evaluación para profesional", padding=10)
        frm_detail.pack(fill="x", padx=8, pady=8)

        self.txt_detail = tk.Text(frm_detail, height=7, wrap="word", font=("Segoe UI", 11), relief="flat", bg="#ffffff")
        self.txt_detail.pack(fill="both", expand=True, padx=4, pady=4)

        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select())
        for widget in (self.ent_day, self.ent_fast, self.ent_post, self.ent_a1c):
            widget.bind("<Return>", lambda e: self._on_enter_from_entry())

    def crear_tablas_db(self):
        """Crear tablas necesarias: participantes y lecturas"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS participantes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                edad INTEGER,
                fecha TEXT,
                lugar TEXT
            )
        """)
        # lecturas: único por participante+día para facilitar actualizaciones
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS lecturas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participante_id INTEGER,
                dia INTEGER,
                fecha TEXT,
                ayuno REAL,
                pos2h REAL,
                a1c REAL,
                UNIQUE(participante_id, dia),
                FOREIGN KEY (participante_id) REFERENCES participantes(id)
            )
        """)
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
        """Inserta o actualiza (upsert) una lectura del participante"""
        if participante_id is None:
            return
        # usar INSERT OR REPLACE aprovechando UNIQUE(participante_id,dia)
        self.cursor.execute("""
            INSERT INTO lecturas (participante_id, dia, fecha, ayuno, pos2h, a1c)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(participante_id, dia) DO UPDATE SET
                fecha=excluded.fecha,
                ayuno=excluded.ayuno,
                pos2h=excluded.pos2h,
                a1c=excluded.a1c
        """, (
            participante_id,
            entry['day'],
            entry['date'],
            entry['fast'],
            entry['post'],
            entry['a1c']
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
        if v <= 120:
            return "Normal"
        if 120 < v <= 140:
            return "Límite alto"
        if 141 <= v <= 199:
            return "Prediabetes"
        return "Diabetes"

    def classify_a1c(self, v):
        if v is None:
            return "-"
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
        for r in self.readings:
            cf = self.classify_fast(r['fast'])
            cp = self.classify_post(r['post'])
            ca = self.classify_a1c(r['a1c']) if r['a1c'] is not None else "-"
            a1c_display = f"{r['a1c']:.2f}" if r['a1c'] is not None else "-"
            self.tree.insert("", "end", values=(r['day'], r['date'], f"{r['fast']:.1f}", f"{r['post']:.1f}", a1c_display, cf, cp, ca))

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

    def evaluate_diabetes_risk(self):
        """Calcula riesgo usando el modelo de regresión logística y muestra reporte."""
        if not self.readings:
            messagebox.showinfo("Evaluación IA", "No hay lecturas disponibles para evaluar.")
            return

        # obtener promedios
        fast_vals = [r['fast'] for r in self.readings]
        post_vals = [r['post'] for r in self.readings]
        a1c_vals = [r['a1c'] for r in self.readings if r['a1c'] is not None]
        avg_fast = statistics.mean(fast_vals)
        avg_post = statistics.mean(post_vals)
        avg_a1c = statistics.mean(a1c_vals) if a1c_vals else 5.0

        # obtener edad del participante si existe
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
            age = simpledialog.askinteger("Edad", "Ingrese la edad del participante (años):", minvalue=1, maxvalue=120)
            if age is None:
                messagebox.showinfo("Evaluación IA", "Edad no proporcionada. Operación cancelada.")
                return

        # preguntar cuidados / estilo de vida
        exercise = simpledialog.askfloat("Ejercicio", "Horas de ejercicio por semana (ej: 2.5):", minvalue=0.0, maxvalue=168.0)
        if exercise is None:
            exercise = 0.0
        diet = simpledialog.askinteger("Dieta", "Califique la calidad de la dieta (1 pobre - 5 excelente):", minvalue=1, maxvalue=5)
        if diet is None:
            diet = 3

        # construir care_score donde 0.0 es excelente y 2.0 es pobre
        care_score = 2.0 - min(2.0, (exercise / 5.0) + ((diet - 1) / 2.0))
        if care_score < 0:
            care_score = 0.0
        if care_score > 2.0:
            care_score = 2.0

        features = [age, avg_fast, avg_post, avg_a1c, care_score]

        if self.diabetes_model is None:
            try:
                self.diabetes_model = train_synthetic_model()
            except Exception:
                messagebox.showerror("IA", "No se pudo inicializar el modelo IA.")
                return

        prob = self.diabetes_model.predict_proba([features])[0]
        timeframe = map_probability_to_timeframe(prob)
        symptoms = possible_symptoms_by_probability(prob)

        report = (
            f"Riesgo estimado de desarrollar diabetes (probabilidad): {prob*100:.1f}%\n"
            f"Estimación de tiempo probable: {timeframe}\n\n"
            "Características usadas para la evaluación:\n"
            f"Edad: {age} años, Glucosa ayuno promedio: {avg_fast:.1f} mg/dL, 2h pos promedio: {avg_post:.1f} mg/dL, A1C promedio: {avg_a1c:.2f}%,\n"
            f"Índice de cuidados (0=mejor,2=peor): {care_score:.2f}\n\n"
            "Síntomas posibles en el futuro:\n"
            + "- " + "\n- ".join(symptoms)
        )

        # mostrar y actualizar resumen
        self.lbl_summary.config(text=report)
        messagebox.showinfo("Evaluación IA - Diabetes", report)

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
            "Evaluar riesgo IA (botón 'Evaluar riesgo IA') - usa modelo entrenado o sintético",
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