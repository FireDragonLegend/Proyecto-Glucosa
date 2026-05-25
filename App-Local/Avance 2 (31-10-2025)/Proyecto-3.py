import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import statistics
import csv
from datetime import datetime

class GlucoseMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor de Glucosa - 1 semana")
        self.root.geometry("760x520")
        self.readings = []  # lista de dicts: {'day':int,'date':str,'fast':float,'post':float,'a1c':float or None}

        self.next_day = 1
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        # estilos más limpios y legibles
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=28)
        style.configure("Primary.TButton", foreground="white", background="#2E8B57", font=("Segoe UI", 11, "bold"))
        style.map("Primary.TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#246b44')])

        # ventana más grande por defecto y fondo suave
        self.root.geometry("1000x660")
        self.root.minsize(920, 600)
        try:
            self.root.configure(bg="#f4f7f6")
        except:
            pass

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=12, pady=12)

        # panel izquierdo: inputs y resumen (más espacioso)
        left = ttk.Frame(paned, width=360)
        paned.add(left, weight=0)

        frm_inputs = ttk.LabelFrame(left, text="Agregar lectura diaria", padding=14)
        frm_inputs.pack(fill="x", padx=8, pady=8)

        ttk.Label(frm_inputs, text="Día (1-7):", style="TLabel").grid(row=0, column=0, sticky="w", pady=4)
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

        # botones principales con mayor tamaño y espaciado
        btn_frame = ttk.Frame(frm_inputs)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(10,0), sticky="we")
        btn_frame.columnconfigure((0,1), weight=1)

        btn_add = ttk.Button(btn_frame, text="Agregar lectura", command=self.add_reading, style="Primary.TButton")
        btn_add.grid(row=0, column=0, padx=(0,6), sticky="we")

        btn_reset = ttk.Button(btn_frame, text="Resetear semana", command=self.reset_week)
        btn_reset.grid(row=0, column=1, sticky="we")

        # fila secundaria de utilidades
        util_frame = ttk.Frame(frm_inputs)
        util_frame.grid(row=5, column=0, columnspan=2, pady=8, sticky="we")
        util_frame.columnconfigure((0,1), weight=1)

        btn_export = ttk.Button(util_frame, text="Exportar CSV", command=self.export_csv)
        btn_export.grid(row=0, column=0, padx=(0,6), sticky="we")
        btn_history = ttk.Button(util_frame, text="Historial detallado", command=self.show_history_window)
        btn_history.grid(row=0, column=1, sticky="we")

        # resumen rápido (para profesional) con fondo claro
        frm_summary = ttk.LabelFrame(left, text="Resumen rápido (para profesional)", padding=10)
        frm_summary.pack(fill="both", expand=False, padx=8, pady=8)

        self.lbl_summary = ttk.Label(frm_summary, text="No hay lecturas aún.", wraplength=320, justify="left")
        self.lbl_summary.pack(fill="both", expand=True, padx=6, pady=6)

        btn_full_summary = ttk.Button(frm_summary, text="Resumen semanal completo", command=self.show_summary)
        btn_full_summary.pack(fill="x", pady=6)

        # botones adicionales para experto
        frm_expert = ttk.LabelFrame(left, text="Herramientas para profesional", padding=8)
        frm_expert.pack(fill="both", expand=False, padx=8, pady=8)

        btn_eval_sel = ttk.Button(frm_expert, text="Evaluar lectura seleccionada", command=self.show_selected_assessment)
        btn_eval_sel.pack(fill="x", pady=4)

        # panel derecho: tabla principal + detalle con mayor contraste
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        header = ttk.Label(right, text="Historial de la semana", style="Header.TLabel")
        header.pack(anchor="w", padx=6, pady=(6,0))

        cols = ("día", "fecha", "ayuno (mg/dL)", "2h pos (mg/dL)", "A1C (%)", "clasif ayuno", "clasif 2h", "clasif A1C")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for c in cols:
            self.tree.heading(c, text=c)
        # columnas más anchas para legibilidad
        self.tree.column("día", width=60, anchor="center")
        self.tree.column("fecha", width=100, anchor="center")
        self.tree.column("ayuno (mg/dL)", width=110, anchor="center")
        self.tree.column("2h pos (mg/dL)", width=120, anchor="center")
        self.tree.column("A1C (%)", width=80, anchor="center")
        self.tree.column("clasif ayuno", width=120, anchor="center")
        self.tree.column("clasif 2h", width=120, anchor="center")
        self.tree.column("clasif A1C", width=120, anchor="center")

        self.tree.pack(fill="both", expand=True, padx=8, pady=6)

        # panel de detalle bajo la tabla con bordes y texto legible
        frm_detail = ttk.LabelFrame(right, text="Detalle y evaluación para profesional", padding=10)
        frm_detail.pack(fill="x", padx=8, pady=8)

        self.txt_detail = tk.Text(frm_detail, height=7, wrap="word", font=("Segoe UI", 11), relief="flat", bg="#ffffff")
        self.txt_detail.pack(fill="both", expand=True, padx=4, pady=4)

        # selección en treeview actualiza detalle breve
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select())

        # BIND: permitir capturar con Enter cuando los campos de entrada están completos
        # se enlaza Return solamente en los Entry relevantes
        for widget in (self.ent_day, self.ent_fast, self.ent_post, self.ent_a1c):
            widget.bind("<Return>", lambda e: self._on_enter_from_entry())

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
            if not (1 <= day <= 7):
                raise ValueError("Día debe estar entre 1 y 7")
        except Exception as e:
            messagebox.showerror("Error", f"Día inválido: {e}")
            return

        try:
            fast = float(self.fast_var.get())
        except:
            messagebox.showerror("Error", "Glucosa en ayunas inválida")
            return
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

        # evitar duplicados de día
        for r in self.readings:
            if r['day'] == day:
                messagebox.showerror("Error", f"Ya existe lectura para el día {day}. Usa otro día o resetea.")
                return

        entry = {
            'day': day,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'fast': fast,
            'post': post,
            'a1c': a1c
        }
        self.readings.append(entry)
        self.readings.sort(key=lambda x: x['day'])
        self._refresh_tree()

        # evaluar con lecturas previas y alertar si corresponde
        self.assess_with_history(entry)

        # incrementar día sugerido hasta 7
        self.next_day = min(7, max(1, self.next_day + 1))
        self.day_var.set(self.next_day)
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
            f"Promedios semana (n={len(self.readings)} lecturas):\n\n"
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
        if messagebox.askyesno("Confirmar", "¿Desea borrar todas las lecturas de la semana?"):
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

def main():
    root = tk.Tk()
    app = GlucoseMonitorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()