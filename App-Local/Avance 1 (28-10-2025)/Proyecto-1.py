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
        style.theme_use('default')

        frm_inputs = ttk.LabelFrame(self.root, text="Agregar lectura diaria", padding=10)
        frm_inputs.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_inputs, text="Día (1-7):").grid(row=0, column=0, sticky="w")
        self.day_var = tk.IntVar(value=self.next_day)
        self.ent_day = ttk.Entry(frm_inputs, textvariable=self.day_var, width=6)
        self.ent_day.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(frm_inputs, text="Glucosa en ayunas (mg/dL):").grid(row=0, column=2, sticky="w")
        self.fast_var = tk.StringVar()
        ttk.Entry(frm_inputs, textvariable=self.fast_var, width=10).grid(row=0, column=3, padx=6)

        ttk.Label(frm_inputs, text="Glucosa 2h pos comida (mg/dL):").grid(row=1, column=0, sticky="w")
        self.post_var = tk.StringVar()
        ttk.Entry(frm_inputs, textvariable=self.post_var, width=10).grid(row=1, column=1, padx=6)

        ttk.Label(frm_inputs, text="A1C (% - opcional):").grid(row=1, column=2, sticky="w")
        self.a1c_var = tk.StringVar()
        ttk.Entry(frm_inputs, textvariable=self.a1c_var, width=10).grid(row=1, column=3, padx=6)

        btn_add = ttk.Button(frm_inputs, text="Agregar lectura", command=self.add_reading)
        btn_add.grid(row=2, column=0, columnspan=2, pady=8, sticky="w")

        btn_reset = ttk.Button(frm_inputs, text="Resetear semana", command=self.reset_week)
        btn_reset.grid(row=2, column=2, pady=8, sticky="w")

        btn_export = ttk.Button(frm_inputs, text="Exportar CSV", command=self.export_csv)
        btn_export.grid(row=2, column=3, pady=8, sticky="w")

        # Treeview con lecturas
        cols = ("día", "fecha", "ayuno (mg/dL)", "2h pos (mg/dL)", "A1C (%)", "clasif ayuno", "clasif 2h", "clasif A1C")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=6)

        frm_summary = ttk.Frame(self.root)
        frm_summary.pack(fill="x", padx=10, pady=6)
        btn_summary = ttk.Button(frm_summary, text="Resumen semanal y clasificación", command=self.show_summary)
        btn_summary.pack(side="left")

        lbl_hint = ttk.Label(frm_summary, text="(A1C opcional; el indicador de resistencia a la insulina es orientativo)")
        lbl_hint.pack(side="left", padx=8)

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
            f"Promedios semana (n={len(self.readings)} lecturas):\n\n"
            f"Glucosa en ayunas promedio: {avg_fast:.1f} mg/dL -> {cf}\n"
            f"Glucosa 2h posprandial promedio: {avg_post:.1f} mg/dL -> {cp}\n"
        )
        if avg_a1c is not None:
            txt += f"A1C promedio: {avg_a1c:.2f}% -> {ca}\n"
        else:
            txt += "A1C promedio: no disponible\n"
        txt += f"\nInterpretación de resistencia a la insulina: {insulin_resistance_hint}\n\n"
        txt += "Nota: Este resultado es orientativo. Consulte a su profesional de salud para diagnóstico."

        messagebox.showinfo("Resumen semanal", txt)

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

def main():
    root = tk.Tk()
    app = GlucoseMonitorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()