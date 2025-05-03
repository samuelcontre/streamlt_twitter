import streamlit as st
import pandas as pd
import re
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configuración de Stopwords y Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\@\w+|\#","", text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def load_data(path='twitter_training.csv'):
    # Leer CSV sin encabezados y asignar columnas
    df = pd.read_csv(path, header=None, names=['ID','Entidad','Label','Texto'])
    # Preprocesar texto
    df['clean_text'] = df['Texto'].astype(str).apply(preprocess_text)
    return df

@st.cache_resource
def train_models(df):
    # Vectorización
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_text'])
    y = df['Label']
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Modelos
    logreg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Entrenar
    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    # Predicciones
    preds_log = logreg.predict(X_test)
    preds_rf = rf.predict(X_test)
    # Métricas
    metrics = {
        'Logistic Regression': {
            'accuracy': accuracy_score(y_test, preds_log),
            'f1': f1_score(y_test, preds_log, average='weighted')
        },
        'Random Forest': {
            'accuracy': accuracy_score(y_test, preds_rf),
            'f1': f1_score(y_test, preds_rf, average='weighted')
        }
    }
    return tfidf, logreg, rf, metrics, X_test, y_test

# Título de la app
st.title(" Predicción de Tuits Virales con IA y Análisis de Sentimiento")

# Cargar datos y entrenar
df = load_data()
tfidf, logreg, rf, metrics, X_test, y_test = train_models(df)

# Visualizaciones EDA
st.subheader(" Distribución de Etiquetas")
counts = df['Label'].value_counts().reset_index()
counts.columns = ['Label','Count']
fig = px.bar(counts, x='Label', y='Count', color='Label', title='Distribución de Sentimientos / Viralidad')
st.plotly_chart(fig)

st.subheader("Nube de Palabras General")
all_text = " ".join(df['clean_text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
st.image(wc.to_array(), use_column_width=True)

st.subheader(" Métricas de Modelos")
for name, m in metrics.items():
    st.write(f"**{name}** - Accuracy: {m['accuracy']:.3f}, F1-score: {m['f1']:.3f}")

# Interfaz de predicción
st.subheader(" Predicción de Viralidad")
model_choice = st.radio("Selecciona el modelo:", ('Logistic Regression','Random Forest'))
user_tweet = st.text_area("Escribe un tweet para predecir:")

if st.button("Predecir") and user_tweet:
    cleaned = preprocess_text(user_tweet)
    vec = tfidf.transform([cleaned])
    model = logreg if model_choice == 'Logistic Regression' else rf
    pred = model.predict(vec)[0]
    st.success(f"La predicción es: {pred}")

# Mostrar reporte de clasificación
st.subheader(" Reporte de Clasificación de Test Set")
report = classification_report(y_test, (logreg.predict(X_test) if model_choice=='Logistic Regression' else rf.predict(X_test)), output_dict=True)
rep_df = pd.DataFrame(report).transpose()
st.dataframe(rep_df)

# Matriz de confusión
st.subheader(" Matriz de Confusión")
cm = confusion_matrix(y_test, logreg.predict(X_test) if model_choice=='Logistic Regression' else rf.predict(X_test))
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
st.dataframe(cm_df)

st.markdown("---")
st.markdown("*App desarrollada en Streamlit y publicada en HuggingFace Spaces* ")
