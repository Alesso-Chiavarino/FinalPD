# 💸 SmartBudget – Asistente Financiero Inteligente

SmartBudget es una aplicación interactiva desarrollada en **Python + Streamlit** que analiza gastos personales, detecta patrones, identifica anomalías y predice el gasto futuro utilizando **algoritmos de Inteligencia Artificial y Machine Learning**.

Este proyecto combina técnicas de **procesamiento de datos, aprendizaje supervisado, aprendizaje no supervisado, NLP (procesamiento de texto)** y visualización interactiva.

---

## 🚀 Características principales

### 🧾 **1. Importación de datos desde Excel**
El usuario puede subir un archivo `.xlsx` con los siguientes campos:
- fecha
- concepto
- monto
- descripcion (opcional)

Incluye un botón para descargar una plantilla base.

---

### 📅 **2. Filtro de fechas inteligente**
La app detecta automáticamente el rango mínimo y máximo de fechas del Excel y ajusta el selector para evitar errores.

Permite visualizar solo el período de análisis deseado.

---

### 📊 **3. Análisis estadístico interactivo**
Incluye visualizaciones generadas con Matplotlib:

- Evolución temporal del gasto  
- Top categorías donde más se gasta  
- Comparación entre meses  
- Distribución por categoría  
- Agrupamientos por día, semana o mes  

---

## 🤖 4. Inteligencia Artificial aplicada

SmartBudget utiliza tres tipos de IA.

### 🔹 A) Machine Learning No Supervisado — KMeans + TF-IDF
Agrupa conceptos similares en categorías inteligentes.

### 🔹 B) Machine Learning Supervisado — RandomForestRegressor
Predice el gasto total del próximo mes.

### 🔹 C) Detección de Anomalías — IsolationForest
Detecta días con gastos fuera de lo común.

---

## 💡 5. Sugerencias automáticas
Genera sugerencias basadas en:

- promedios históricos
- desvíos significativos
- impacto porcentual por categoría

---

## 🎨 6. Interfaz intuitiva (Streamlit)
Organizada en:

- Gráficos  
- Panel de predicción IA  
- Tabs: Detalles, Categorías, Anomalías, Sugerencias  
- Exportación de CSV  

---

# 📂 Estructura del proyecto

```
SmartBudget/
│── app.py                
│── modelo.py             
│── utils.py              
│── gastos.xlsx           
│── requirements.txt      
│── README.md             
```

---

# 📦 Instalación

Clonar el repositorio:

git clone https://github.com/usuario/SmartBudget.git
cd SmartBudget

Instalar dependencias:

pip install -r requirements.txt

---

# ▶️ Ejecución del proyecto

streamlit run app.py

Abrirá en:

http://localhost:8501

---

# 📝 Formato del archivo Excel

| fecha       | concepto     | monto | descripcion |
|-------------|--------------|-------|-------------|
| 2024-01-02  | supermercado | 4200  | compra mes  |
| 2024-01-03  | uber         | 950   | trabajo     |

---

# 👤 Autor
Desarrollado por: **[Tu nombre]**
Materia: **Programación Declarativa – Final Python**
