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

class MedicoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación Médica - Evaluación de Pacientes")
        self.root.geometry("1300x850")
        self.root.minsize(1200, 750)
        self.readings = []

        self.next_day = 1
        self.default_a1c = None

        # Database
        self.db_path = "Proyecto.db"  # Compartir BD
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.current_participant_id = None
        self.participantes = []
        self.cargar_participantes_desde_db()

        # modelo IA avanzado
        self.diabetes_model = None
        self.model_loading = False
        self.model_status = "No cargado"

        self._build_ui()

        # cargar modelo en background
        self.model_thread = threading.Thread(target=self._background_load_model, daemon=True)
        self.model_thread.start()

    def _background_load_model(self):
        try:
            self.model_loading = True
            self.model_status = "Cargando modelo..."
            print("Cargando modelo IA...")

            self.diabetes_model = self.load_or_train_model()

            if self.diabetes_model is not None:
                self.model_status = "Modelo listo"
                print("Modelo IA listo")
            else:
                self.model_status = "Error al cargar modelo"
                print("Error al cargar modelo IA")

        except Exception as e:
            self.model_status = f"Error: {str(e)}"
            print(f"Error en modelo: {e}")
        finally:
            self.model_loading = False

    def load_or_train_model(self):
        import os
        model_path = "diabetes_model.pkl"

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Modelo cargado desde archivo.")
                return model
            except Exception as e:
                print(f"Error cargando modelo: {e}")
                return None
        else:
            print("No hay modelo guardado.")
            return None

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("TLabel", font=("Segoe UI", 10), background="white", foreground="#1f2937")
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="white", background="white")
        style.configure("TFrame", background="white")
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="white", foreground="#1e40af")
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=24, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", background="#f8fafc", foreground="#1e40af", font=("Segoe UI", 9, "bold"))
        style.configure("Card.TFrame", background="white", relief="raised", borderwidth=1)
        style.configure("TLabelframe", background="white", foreground="#1f2937")
        style.configure("TLabelframe.Label", background="white", foreground="#000000", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[('selected', '#bfdbfe')], foreground=[('selected', '#000000')])
        style.configure("Primary.TButton", foreground="white", background="#10b981", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", foreground=[('active', 'white')], background=[('active', '#059669')])
        style.configure("Danger.TButton", foreground="white", background="#ef4444", font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton", foreground=[('active', 'white')], background=[('active', '#dc2626')])

        banner = ttk.Label(self.root, text="🏥 Aplicación Médica - Evaluación de Diabetes", style="Header.TLabel")
        banner.pack(fill="x", pady=(3, 10))

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.configure(style="TFrame")
        paned.pack(fill="both", expand=True, padx=12, pady=12)

        left_container = ttk.Frame(paned, width=450)
        left_container.configure(style="Card.TFrame")
        paned.add(left_container, weight=0)

        canvas_left = tk.Canvas(left_container, highlightthickness=0)
        vscroll = ttk.Scrollbar(left_container, orient="vertical", command=canvas_left.yview)
        canvas_left.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas_left.pack(side="left", fill="both", expand=True)

        left = ttk.Frame(canvas_left, width=430)
        left.configure(style="Card.TFrame")
        canvas_left.create_window((0, 0), window=left, anchor="nw")

        def _left_configure(event):
            canvas_left.configure(scrollregion=canvas_left.bbox("all"))

        left.bind("<Configure>", _left_configure)

        # Panel de selección de paciente
        frm_paciente = ttk.LabelFrame(left, text="Selección de Paciente", padding=12)
        frm_paciente.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_paciente, text="Seleccionar paciente:").grid(row=0, column=0, sticky="w")
        self.paciente_var = tk.StringVar()
        self.paciente_cb = ttk.Combobox(frm_paciente, textvariable=self.paciente_var, state="readonly", width=35)
        self.paciente_cb.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        self.paciente_cb.bind("<<ComboboxSelected>>", lambda e: self.on_paciente_selected())

        # Mostrar info del paciente
        self.info_paciente_label = ttk.Label(frm_paciente, text="Seleccione un paciente para ver información.", wraplength=350, justify="left")
        self.info_paciente_label.grid(row=1, column=0, columnspan=2, pady=6, sticky="w")

        # Corrección de lectura (seleccione día para cargar valores existentes)
        frm_lectura = ttk.LabelFrame(left, text="Corrección de lectura", padding=12)
        frm_lectura.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_lectura, text="Día (1-90):").grid(row=0, column=0, sticky="w", pady=4)
        self.day_var = tk.IntVar(value=self.next_day)
        # Usamos Combobox para que se pueda seleccionar rápidamente un día con datos guardados
        self.day_cb = ttk.Combobox(frm_lectura, textvariable=self.day_var, values=list(range(1, 91)), width=6)
        self.day_cb.grid(row=0, column=1, padx=6, pady=4, sticky="w")
        self.day_cb.bind("<<ComboboxSelected>>", lambda e: self._load_reading_for_day(self.day_var.get()))
        self.day_cb.bind("<FocusOut>", lambda e: self._load_reading_for_day(self.day_var.get()))
        self.day_cb.bind("<Return>", lambda e: self._load_reading_for_day(self.day_var.get()))

        ttk.Label(frm_lectura, text="Glucosa ayunas (mg/dL):").grid(row=1, column=0, sticky="w", pady=4)
        self.fast_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.fast_var, width=12).grid(row=1, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_lectura, text="Glucosa 2h post (mg/dL):").grid(row=2, column=0, sticky="w", pady=4)
        self.post_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.post_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_lectura, text="A1C (%):").grid(row=3, column=0, sticky="w", pady=4)
        self.a1c_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.a1c_var, width=12).grid(row=3, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_lectura, text="LDL (mg/dL):").grid(row=4, column=0, sticky="w", pady=4)
        self.ldl_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.ldl_var, width=12).grid(row=4, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_lectura, text="HDL (mg/dL):").grid(row=5, column=0, sticky="w", pady=4)
        self.hdl_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.hdl_var, width=12).grid(row=5, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_lectura, text="Triglicéridos (mg/dL):").grid(row=6, column=0, sticky="w", pady=4)
        self.trig_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.trig_var, width=12).grid(row=6, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_lectura, text="Colesterol Total (mg/dL):").grid(row=7, column=0, sticky="w", pady=4)
        self.total_chol_var = tk.StringVar()
        ttk.Entry(frm_lectura, textvariable=self.total_chol_var, width=12).grid(row=7, column=1, padx=6, pady=4, sticky="w")

        btn_add = ttk.Button(frm_lectura, text="Guardar corrección", command=self.add_reading, style="Primary.TButton")
        btn_add.grid(row=8, column=0, columnspan=2, pady=6, sticky="we")

        # Resumen rápido
        frm_resumen = ttk.LabelFrame(left, text="Resumen del Estado del Paciente", padding=12)
        frm_resumen.pack(fill="both", expand=False, padx=10, pady=8)

        self.lbl_resumen = ttk.Label(frm_resumen, text="Seleccione un paciente para ver resumen.", wraplength=350, justify="left")
        self.lbl_resumen.pack(fill="both", expand=True, padx=6, pady=6)

        # Herramientas para profesional
        frm_herramientas = ttk.LabelFrame(left, text="Herramientas Profesionales", padding=12)
        frm_herramientas.pack(fill="both", expand=False, padx=10, pady=8)

        btn_eval = ttk.Button(frm_herramientas, text="Evaluar con IA Avanzada", command=self.evaluate_diabetes_risk)
        btn_eval.pack(fill="x", pady=2)

        self.model_status_label = ttk.Label(frm_herramientas, text="Estado IA: Inicializando...", foreground="#666666", font=("Segoe UI", 8))
        self.model_status_label.pack(fill="x", pady=(2, 0))

        self._update_model_status()

        right = ttk.Frame(paned)
        right.configure(style="Card.TFrame")
        paned.add(right, weight=1)

        header = ttk.Label(right, text="Historial de Lecturas del Paciente", style="Header.TLabel")
        header.pack(anchor="w", padx=10, pady=(12, 6))

        cols = ("día", "fecha", "ayuno", "2h pos", "A1C", "LDL", "HDL", "Trig", "Total Chol", "clasif ayuno", "clasif 2h", "clasif A1C")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
        self.tree.tag_configure('oddrow', background='#f9fafb')
        self.tree.tag_configure('evenrow', background='white')
        for c in cols:
            self.tree.heading(c, text=c.title())
            if c in ["día", "fecha"]:
                self.tree.column(c, width=60, anchor="center")
            else:
                self.tree.column(c, width=70, anchor="center")

        self.tree.pack(fill="both", expand=True, padx=10, pady=8)

        # Detalles y evaluación
        frm_detalle = ttk.LabelFrame(right, text="Detalles y Notas del Profesional", padding=12)
        frm_detalle.pack(fill="x", padx=10, pady=8)

        self.txt_detalle = tk.Text(frm_detalle, height=8, wrap="word", font=("Segoe UI", 10), relief="flat", bg="#ffffff")
        self.txt_detalle.pack(fill="both", expand=True, padx=6, pady=6)

        btn_guardar_nota = ttk.Button(frm_detalle, text="Guardar Nota", command=self.guardar_nota)
        btn_guardar_nota.pack(fill="x", pady=6)

        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select())

        self._refresh_paciente_combobox()

    def _update_model_status(self):
        if hasattr(self, 'model_status_label'):
            if self.model_loading:
                self.model_status_label.config(text="Estado IA: Cargando...", foreground="#ff6b35")
            elif self.diabetes_model is not None:
                self.model_status_label.config(text="Estado IA: Listo ✓", foreground="#10b981")
            else:
                self.model_status_label.config(text="Estado IA: Error ✗", foreground="#ef4444")

        self.root.after(2000, self._update_model_status)

    def cargar_participantes_desde_db(self):
        self.participantes.clear()
        self.cursor.execute("SELECT id, nombre FROM participantes ORDER BY nombre")
        filas = self.cursor.fetchall()
        for fila in filas:
            self.participantes.append((fila[0], fila[1]))

    def cargar_lecturas_desde_db(self, participante_id):
        self.readings.clear()
        if participante_id is None:
            return
        self.cursor.execute("""
            SELECT dia, fecha, ayuno, pos2h, a1c, ldl, hdl, triglycerides, total_cholesterol
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
                'a1c': f[4],
                'ldl': f[5],
                'hdl': f[6],
                'triglycerides': f[7],
                'total_cholesterol': f[8]
            }
            self.readings.append(entry)
        self.readings.sort(key=lambda x: x['day'])

    def _load_reading_for_day(self, day):
        """Carga en los campos los valores ya guardados para el día seleccionado."""
        try:
            day_int = int(day)
        except Exception:
            return
        existing = next((r for r in self.readings if r['day'] == day_int), None)
        if existing:
            # Mostrar valores existentes para edición
            self.fast_var.set(f"{existing['fast']:.1f}" if existing['fast'] is not None else "")
            self.post_var.set(f"{existing['post']:.1f}" if existing['post'] is not None else "")
            self.a1c_var.set(f"{existing['a1c']:.1f}" if existing['a1c'] is not None else "")
            self.ldl_var.set(f"{existing['ldl']:.1f}" if existing.get('ldl') is not None else "")
            self.hdl_var.set(f"{existing['hdl']:.1f}" if existing.get('hdl') is not None else "")
            self.trig_var.set(f"{existing['triglycerides']:.1f}" if existing.get('triglycerides') is not None else "")
            self.total_chol_var.set(f"{existing['total_cholesterol']:.1f}" if existing.get('total_cholesterol') is not None else "")
        else:
            # Limpiar valores para nuevos días
            self._clear_inputs()

    def _refresh_paciente_combobox(self):
        self.paciente_cb['values'] = [p[1] for p in self.participantes]
        if self.participantes:
            self.paciente_cb.current(0)
            self.on_paciente_selected()

    def on_paciente_selected(self):
        selected = self.paciente_cb.get()
        paciente = next((p for p in self.participantes if p[1] == selected), None)
        if paciente:
            self.current_participant_id = paciente[0]
            self.cargar_lecturas_desde_db(self.current_participant_id)
            self._refresh_tree()
            self.mostrar_info_paciente(paciente[0])
            self.mostrar_resumen_paciente()
            # Al seleccionar paciente, cargar lectura del día seleccionado (si existe)
            self._load_reading_for_day(self.day_var.get())

    def mostrar_info_paciente(self, paciente_id):
        self.cursor.execute("SELECT nombre, edad, fecha, lugar, fecha_nacimiento FROM participantes WHERE id = ?", (paciente_id,))
        fila = self.cursor.fetchone()
        if fila:
            nombre, edad, fecha_ingreso, lugar, fecha_nac = fila
            info = f"Nombre: {nombre}\nEdad: {edad}\nFecha de Ingreso: {fecha_ingreso}\nFecha de Nacimiento: {fecha_nac}\nLugar: {lugar}"
            self.info_paciente_label.config(text=info)

    def mostrar_resumen_paciente(self):
        if not self.readings:
            self.lbl_resumen.config(text="No hay lecturas disponibles.")
            return

        # Calcular promedios
        fast_vals = [r['fast'] for r in self.readings if r['fast'] > 0]
        post_vals = [r['post'] for r in self.readings if r['post'] > 0]
        a1c_vals = [r['a1c'] for r in self.readings if r['a1c'] is not None]
        ldl_vals = [r['ldl'] for r in self.readings if r['ldl'] is not None]
        hdl_vals = [r['hdl'] for r in self.readings if r['hdl'] is not None]
        trig_vals = [r['triglycerides'] for r in self.readings if r['triglycerides'] is not None]

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

        self.lbl_resumen.config(text=resumen)

    def add_reading(self):
        # Similar a Proyecto.py, pero sin validaciones mínimas estrictas
        try:
            day = int(self.day_var.get())
            if not (1 <= day <= 90):
                raise ValueError("Día debe estar entre 1 y 90")
        except Exception as e:
            messagebox.showerror("Error", f"Día inválido: {e}")
            return

        fast = None
        if self.fast_var.get().strip() != "":
            try:
                fast = float(self.fast_var.get())
            except:
                messagebox.showerror("Error", "Glucosa en ayunas inválida")
                return

        post = None
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

        ldl = None
        if self.ldl_var.get().strip() != "":
            try:
                ldl = float(self.ldl_var.get())
            except:
                messagebox.showerror("Error", "LDL inválido")
                return

        hdl = None
        if self.hdl_var.get().strip() != "":
            try:
                hdl = float(self.hdl_var.get())
            except:
                messagebox.showerror("Error", "HDL inválido")
                return

        trig = None
        if self.trig_var.get().strip() != "":
            try:
                trig = float(self.trig_var.get())
            except:
                messagebox.showerror("Error", "Triglicéridos inválidos")
                return

        total_chol = None
        if self.total_chol_var.get().strip() != "":
            try:
                total_chol = float(self.total_chol_var.get())
            except:
                messagebox.showerror("Error", "Colesterol Total inválido")
                return

        existing = next((r for r in self.readings if r['day'] == day), None)
        if existing:
            existing['fast'] = fast if fast is not None else existing['fast']
            existing['post'] = post if post is not None else existing['post']
            existing['a1c'] = a1c if a1c is not None else existing['a1c']
            existing['ldl'] = ldl if ldl is not None else existing.get('ldl')
            existing['hdl'] = hdl if hdl is not None else existing.get('hdl')
            existing['triglycerides'] = trig if trig is not None else existing.get('triglycerides')
            existing['total_cholesterol'] = total_chol if total_chol is not None else existing.get('total_cholesterol')
            existing['date'] = datetime.now().strftime("%Y-%m-%d")
            if self.current_participant_id:
                self.guardar_lectura_en_db(self.current_participant_id, existing)
            self._refresh_tree()
            messagebox.showinfo("Actualizado", f"Lectura día {day} actualizada.")
            self._clear_inputs()
            return

        entry = {
            'day': day,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'fast': fast if fast is not None else 0.0,
            'post': post if post is not None else 0.0,
            'a1c': a1c,
            'ldl': ldl,
            'hdl': hdl,
            'triglycerides': trig,
            'total_cholesterol': total_chol
        }
        self.readings.append(entry)
        self.readings.sort(key=lambda x: x['day'])

        if self.current_participant_id:
            self.guardar_lectura_en_db(self.current_participant_id, entry)
        else:
            messagebox.showwarning("Sin paciente", "No hay paciente seleccionado.")

        self._refresh_tree()
        self.mostrar_resumen_paciente()

        self.next_day = min(90, max(1, self.next_day + 1))
        self.day_var.set(self.next_day)
        self._clear_inputs()

    def guardar_lectura_en_db(self, participante_id, entry):
        if participante_id is None:
            return
        ldl = entry.get('ldl', None)
        hdl = entry.get('hdl', None)
        triglycerides = entry.get('triglycerides', None)
        total_cholesterol = entry.get('total_cholesterol', None)

        self.cursor.execute("""
            INSERT OR REPLACE INTO lecturas 
            (participante_id, dia, fecha, ayuno, pos2h, a1c, ldl, hdl, triglycerides, total_cholesterol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (participante_id, entry['day'], entry['date'], entry['fast'], entry['post'], entry['a1c'], ldl, hdl, triglycerides, total_cholesterol))
        self.conn.commit()

    def _clear_inputs(self):
        self.fast_var.set("")
        self.post_var.set("")
        self.a1c_var.set("")
        self.ldl_var.set("")
        self.hdl_var.set("")
        self.trig_var.set("")
        self.total_chol_var.set("")

    def _refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, r in enumerate(self.readings):
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            cf = self.classify_fast(r['fast'])
            cp = self.classify_post(r['post'])
            ca = self.classify_a1c(r['a1c']) if r['a1c'] else ""
            self.tree.insert("", "end", values=(
                r['day'], r['date'], f"{r['fast']:.1f}" if r['fast'] > 0 else "",
                f"{r['post']:.1f}" if r['post'] > 0 else "",
                f"{r['a1c']:.1f}" if r['a1c'] else "",
                f"{r['ldl']:.1f}" if r['ldl'] else "",
                f"{r['hdl']:.1f}" if r['hdl'] else "",
                f"{r['triglycerides']:.1f}" if r['triglycerides'] else "",
                f"{r['total_cholesterol']:.1f}" if r['total_cholesterol'] else "",
                cf, cp, ca
            ), tags=(tag,))

    def classify_fast(self, v):
        if v >= 126:
            return "Diabetes"
        elif v >= 100:
            return "Prediabetes"
        else:
            return "Normal"

    def classify_post(self, v):
        if v >= 200:
            return "Diabetes"
        elif v >= 140:
            return "Prediabetes"
        else:
            return "Normal"

    def classify_a1c(self, v):
        if v >= 6.5:
            return "Diabetes"
        elif v >= 5.7:
            return "Prediabetes"
        else:
            return "Normal"

    def _on_tree_select(self):
        selected = self.tree.selection()
        if selected:
            item = self.tree.item(selected[0])
            values = item['values']
            detalle = f"Día: {values[0]}\nFecha: {values[1]}\nGlucosa Ayunas: {values[2]}\nGlucosa 2h: {values[3]}\nA1C: {values[4]}\nLDL: {values[5]}\nHDL: {values[6]}\nTriglicéridos: {values[7]}\nTotal Colesterol: {values[8]}"
            self.txt_detalle.delete(1.0, tk.END)
            self.txt_detalle.insert(tk.END, detalle)

    def guardar_nota(self):
        if not self.current_participant_id:
            messagebox.showerror("Error", "Seleccione un paciente primero.")
            return
        nota = self.txt_detalle.get(1.0, tk.END).strip()
        if not nota:
            messagebox.showerror("Error", "La nota no puede estar vacía.")
            return
        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Crear tabla notas si no existe
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS notas_paciente (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participante_id INTEGER,
                nota TEXT,
                fecha TEXT,
                FOREIGN KEY (participante_id) REFERENCES participantes(id)
            )
        """)
        self.cursor.execute("INSERT INTO notas_paciente (participante_id, nota, fecha) VALUES (?, ?, ?)", (self.current_participant_id, nota, fecha))
        self.conn.commit()
        messagebox.showinfo("Nota", "Nota guardada correctamente.")

    def evaluate_diabetes_risk(self):
        if not self.current_participant_id or not self.diabetes_model:
            messagebox.showerror("Error", "Seleccione un paciente y asegúrese de que el modelo IA esté listo.")
            return

        # Obtener datos del paciente
        self.cursor.execute("SELECT edad, fecha_nacimiento FROM participantes WHERE id = ?", (self.current_participant_id,))
        fila = self.cursor.fetchone()
        if not fila:
            messagebox.showerror("Error", "Datos del paciente no encontrados.")
            return
        edad, fecha_nac = fila

        # Calcular edad actual si fecha_nac disponible
        if fecha_nac:
            try:
                nac = datetime.strptime(fecha_nac, "%Y-%m-%d")
                edad_actual = (datetime.now() - nac).days // 365
            except:
                edad_actual = edad
        else:
            edad_actual = edad

        # Preguntar datos adicionales con ventana
        win = tk.Toplevel(self.root)
        win.title("Evaluación IA - Datos Adicionales")
        win.geometry("500x600")

        ttk.Label(win, text="Datos para evaluación IA:").pack(pady=10)

        # Preguntas
        questions = [
            ("Peso (kg):", "float"),
            ("Altura (cm):", "float"),
            ("Presión Sistólica:", "int"),
            ("Presión Diastólica:", "int"),
            ("¿Antecedentes familiares? (1=Sí, 0=No):", "int"),
            ("Ejercicio semanal (minutos):", "int"),
            ("Calidad dieta (1-5, 5=excelente):", "int"),
        ]

        entries = {}
        for q, typ in questions:
            frame = ttk.Frame(win)
            frame.pack(fill="x", padx=10, pady=2)
            ttk.Label(frame, text=q).pack(side="left")
            var = tk.StringVar()
            ent = ttk.Entry(frame, textvariable=var, width=10)
            ent.pack(side="right")
            entries[q] = (var, typ)

        def submit():
            try:
                peso = float(entries["Peso (kg):"][0].get())
                altura = float(entries["Altura (cm):"][0].get()) / 100  # a metros
                bmi = peso / (altura ** 2)
                systolic = int(entries["Presión Sistólica:"][0].get())
                diastolic = int(entries["Presión Diastólica:"][0].get())
                family = int(entries["¿Antecedentes familiares? (1=Sí, 0=No):"][0].get())
                exercise = int(entries["Ejercicio semanal (minutos):"][0].get())
                diet = int(entries["Calidad dieta (1-5, 5=excelente):"][0].get())

                # Obtener promedios de lecturas
                if self.readings:
                    avg_fast = statistics.mean([r['fast'] for r in self.readings if r['fast'] > 0]) if any(r['fast'] > 0 for r in self.readings) else 100
                    avg_post = statistics.mean([r['post'] for r in self.readings if r['post'] > 0]) if any(r['post'] > 0 for r in self.readings) else 140
                    avg_a1c = statistics.mean([r['a1c'] for r in self.readings if r['a1c']]) if any(r['a1c'] for r in self.readings) else 5.5
                    avg_ldl = statistics.mean([r['ldl'] for r in self.readings if r['ldl']]) if any(r['ldl'] for r in self.readings) else 100
                    avg_hdl = statistics.mean([r['hdl'] for r in self.readings if r['hdl']]) if any(r['hdl'] for r in self.readings) else 50
                    avg_trig = statistics.mean([r['triglycerides'] for r in self.readings if r['triglycerides']]) if any(r['triglycerides'] for r in self.readings) else 150
                    avg_total = statistics.mean([r['total_cholesterol'] for r in self.readings if r['total_cholesterol']]) if any(r['total_cholesterol'] for r in self.readings) else 200
                else:
                    avg_fast = 100
                    avg_post = 140
                    avg_a1c = 5.5
                    avg_ldl = 100
                    avg_hdl = 50
                    avg_trig = 150
                    avg_total = 200

                features = [edad_actual, avg_fast, avg_post, avg_a1c, bmi, avg_ldl, avg_hdl, avg_trig, avg_total, systolic, diastolic, 0, family, exercise, diet]  # care_score=0

                prob, stage, time_to_diabetes, risk_factors, recommendations = assess_diabetes_risk_comprehensive(features, self.diabetes_model)

                avg_glucose = max(avg_fast, avg_post)
                timeframe = map_probability_to_timeframe(prob, edad_actual, avg_glucose, avg_ldl, avg_hdl, stage)

                symptoms = possible_symptoms_by_probability(prob, stage, edad_actual)

                # Recomendaciones de ejercicios
                exercise_rec = "Realice al menos 150 minutos de ejercicio moderado por semana, como caminar, nadar o ciclismo."

                # Horario de medicamentos (simulado)
                med_schedule = "Tomar medicamentos en ayunas por la mañana, y después de comidas según prescripción médica."

                # Rango de años para prediabetes/diabetes
                age_range = f"Posible prediabetes entre {edad_actual + 5}-{edad_actual + 10} años, diabetes entre {edad_actual + 10}-{edad_actual + 20} años basado en factores de riesgo."

                report = f"Evaluación IA:\n\nProbabilidad: {prob:.2f}\nEtapa: {stage}\nTimeframe: {timeframe}\n\nFactores de Riesgo:\n" + "\n".join(risk_factors) + "\n\nRecomendaciones:\n" + "\n".join(recommendations) + "\n\nEjercicios: " + exercise_rec + "\n\nSíntomas Posibles:\n" + "\n".join(symptoms) + "\n\nHorario Medicamentos: " + med_schedule + "\n\nRango Edad Riesgo: " + age_range

                self._show_report_window(report)
                win.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Error en evaluación: {e}")

        ttk.Button(win, text="Evaluar", command=submit).pack(pady=10)

    def _show_report_window(self, report_text):
        report_win = tk.Toplevel(self.root)
        report_win.title("Reporte de Evaluación IA")
        report_win.geometry("600x700")

        txt = tk.Text(report_win, wrap="word", font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert(tk.END, report_text)

        btn_copy = ttk.Button(report_win, text="Copiar al Portapapeles", command=lambda: self._copy_to_clipboard(report_text))
        btn_copy.pack(side="left", padx=10, pady=10)

        btn_save = ttk.Button(report_win, text="Guardar Reporte", command=lambda: self._save_report(report_text))
        btn_save.pack(side="right", padx=10, pady=10)

    def _copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copiado", "Reporte copiado al portapapeles.")

    def _save_report(self, text):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as f:
                f.write(text)
            messagebox.showinfo("Guardado", "Reporte guardado.")

def main():
    root = tk.Tk()
    app = MedicoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()