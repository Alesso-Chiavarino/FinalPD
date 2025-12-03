import pandas as pd
import numpy as np

# 🔹 Fechas diarias (primer semestre de 2025)
rng = pd.date_range("2025-01-01", "2025-06-30", freq="D")

# 🔹 Conceptos base (pueden repetirse con distinta descripción)
conceptos = [
    "Supermercado", "Transporte", "Gasolina", "Restaurante", "Internet",
    "Servicios", "Cine", "Ropa", "Farmacia", "Café", "Suscripciones", "Mantenimiento"
]

np.random.seed(42)
filas = []

for fecha in rng:
    # Cada día puede tener entre 0 y 4 gastos
    n = np.random.randint(0, 5)
    for _ in range(n):
        concepto = np.random.choice(conceptos)
        # monto aleatorio con distribución Gamma (más realista para gastos)
        monto = round(np.random.gamma(2.5, 15), 2)
        descripcion = f"Gasto en {concepto.lower()} - ticket #{np.random.randint(1000,9999)}"
        filas.append([fecha, concepto, descripcion, monto])

# 🔹 Crear DataFrame
df = pd.DataFrame(filas, columns=["fecha", "concepto", "descripcion", "monto"])

# 🔹 Guardar en Excel
df.to_excel("gastos.xlsx", index=False)
print(f"✅ Archivo 'gastos.xlsx' creado correctamente con {len(df)} registros.")