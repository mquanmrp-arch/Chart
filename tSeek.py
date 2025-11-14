import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io

# Configuración de página
st.set_page_config(
    page_title="📈 Trading Pattern Predictor",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📈 Predictor de Tendencias - Análisis Técnico con IA")
st.markdown("---")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_tipo = st.selectbox(
        "Tipo de Modelo",
        ["Binario (Alcista/Bajista)", "Multi-clase (Patrones)"]
    )
    
    st.markdown("---")
    st.subheader("📤 Cargar Modelo")
    modelo_file = st.file_uploader(
        "Sube tu modelo .h5",
        type=['h5'],
        help="Modelo entrenado de TensorFlow/Keras"
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Entrena tu modelo en Google Colab y descarga el archivo .h5")

# Área principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Subir Imagen del Gráfico")
    uploaded_image = st.file_uploader(
        "Selecciona una imagen (150x150 px recomendado)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Imagen cargada", use_container_width=True)

with col2:
    st.subheader("🎯 Resultados de Predicción")
    
    if uploaded_image and modelo_file:
        try:
            # Cargar modelo
            with st.spinner("Cargando modelo..."):
                model = tf.keras.models.load_model(modelo_file)
            
            # Preprocesar imagen
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predicción
            with st.spinner("Analizando..."):
                predictions = model.predict(img_array, verbose=0)
            
            # Resultados según tipo de modelo
            if modelo_tipo == "Binario (Alcista/Bajista)":
                prob_alcista = predictions[0][0]
                prob_bajista = 1 - prob_alcista
                
                # Determinar tendencia
                if prob_alcista > prob_bajista:
                    tendencia = "📈 ALCISTA"
                    confianza = prob_alcista
                    color = "green"
                else:
                    tendencia = "📉 BAJISTA"
                    confianza = prob_bajista
                    color = "red"
                
                # Mostrar resultado destacado
                st.markdown(f"### {tendencia}")
                st.markdown(f"**Confianza:** {confianza:.1%}")
                
                # Barra de progreso
                st.progress(confianza)
                
                # Detalles
                with st.expander("Ver probabilidades detalladas"):
                    st.metric("Probabilidad Alcista", f"{prob_alcista:.2%}")
                    st.metric("Probabilidad Bajista", f"{prob_bajista:.2%}")
                
                # Gráfico de barras
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(['Bajista', 'Alcista'], 
                       [prob_bajista, prob_alcista],
                       color=['red', 'green'])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilidad')
                ax.set_title('Distribución de Probabilidades')
                st.pyplot(fig)
                
            else:  # Multi-clase
                # Nombres de patrones (ajusta según tus clases)
                patrones = [
                    "Bandera Alcista", "Bandera Bajista",
                    "Triángulo Ascendente", "Triángulo Descendente",
                    "Hombro-Cabeza-Hombro", "Doble Techo",
                    "Cuña Ascendente", "Cuña Descendente"
                ]
                
                # Asegurar que hay suficientes nombres
                if len(patrones) < len(predictions[0]):
                    patrones = [f"Patrón {i+1}" for i in range(len(predictions[0]))]
                
                # Ordenar por probabilidad
                indices_ordenados = np.argsort(predictions[0])[::-1]
                
                # Patrón más probable
                patron_predicho = patrones[indices_ordenados[0]]
                confianza_max = predictions[0][indices_ordenados[0]]
                
                st.markdown(f"### 🎯 {patron_predicho}")
                st.markdown(f"**Confianza:** {confianza_max:.1%}")
                st.progress(confianza_max)
                
                # Top 3 patrones
                with st.expander("Ver Top 3 Patrones"):
                    for i in range(min(3, len(indices_ordenados))):
                        idx = indices_ordenados[i]
                        st.metric(
                            patrones[idx],
                            f"{predictions[0][idx]:.2%}"
                        )
                
                # Gráfico de barras
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(patrones))
                colors = ['green' if i == indices_ordenados[0] else 'skyblue' 
                         for i in range(len(patrones))]
                ax.barh(y_pos, predictions[0], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(patrones)
                ax.set_xlabel('Probabilidad')
                ax.set_title('Probabilidades por Patrón')
                ax.set_xlim(0, 1)
                st.pyplot(fig)
            
            st.success("✅ Análisis completado exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error al procesar: {str(e)}")
            st.info("Verifica que el modelo y la imagen sean compatibles")
    
    elif not modelo_file:
        st.warning("⚠️ Por favor, carga un modelo primero")
    elif not uploaded_image:
        st.info("📤 Sube una imagen para comenzar el análisis")

# Footer con instrucciones
st.markdown("---")
with st.expander("📚 ¿Cómo usar esta aplicación?"):
    st.markdown("""
    ### Pasos para usar el predictor:
    
    1. **Entrenar tu modelo en Google Colab:**
       - Usa el código de entrenamiento proporcionado
       - Descarga el archivo `.h5` generado
    
    2. **Cargar el modelo:**
       - En el sidebar, sube el archivo `.h5`
       - Selecciona el tipo de modelo (Binario o Multi-clase)
    
    3. **Subir gráfico:**
       - Carga una imagen del gráfico de velas (150x150 px recomendado)
       - La app redimensionará automáticamente si es necesario
    
    4. **Ver resultados:**
       - La predicción se mostrará automáticamente
       - Puedes ver probabilidades detalladas y gráficos
    
    ### 📊 Tipos de análisis:
    - **Binario:** Determina si la tendencia es alcista o bajista
    - **Multi-clase:** Identifica patrones chartistas específicos
    """)

st.markdown("---")
st.caption("🔧 Desarrollado con TensorFlow + Streamlit | 📈 Trading con IA")
