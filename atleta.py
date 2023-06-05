import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_super_athlete_model():
    # importando dados
    df = pd.read_csv('dados.csv', decimal=",")

    # Remova a coluna de nomes, pois não é uma variável numérica
    df_numeric = df.drop('nome', axis=1)

    # Escalonar os dados para ter média 0 e variância 1
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)  # Ajuste o número de clusters conforme necessário
    clusters = kmeans.fit_predict(df_scaled)

    # Adicionar os clusters ao DataFrame original
    df['Cluster'] = clusters

    # Calcule a média de cada cluster
    cluster_means = df.groupby('Cluster').mean(numeric_only=True)

    # Encontre o cluster com a maior média (supondo que valores mais altos são melhores para todas as variáveis)
    best_cluster = cluster_means.mean(axis=1).idxmax()

    return scaler, kmeans, best_cluster

def predict_super_athlete(new_data, scaler, kmeans, best_cluster):
    # Remova a coluna 'nome', pois não é uma variável numérica
    new_data_numeric = new_data.drop('nome', axis=1)

    # Escalonar os dados para ter média 0 e variância 1 usando o scaler já treinado
    new_data_scaled = scaler.transform(new_data_numeric)

    # Aplicar K-Means usando o modelo já treinado
    new_clusters = kmeans.predict(new_data_scaled)

    # Adicionar os clusters ao DataFrame original
    new_data['Cluster'] = new_clusters

    # Verificar se os novos atletas são super atletas
    new_data['is_super_athlete'] = new_data['Cluster'] == best_cluster

    return new_data

# Treinar o modelo de atleta super
scaler, kmeans, best_cluster = train_super_athlete_model()

# Criar a interface do Streamlit
st.title("Seleção de Super Atletas")
st.write("Insira as informações dos novos atletas:")

# Campos de entrada para as informações dos atletas
nome = st.text_input("Nome do Atleta")
idade = st.number_input("Idade", min_value=0)
estatura = st.number_input("Estatura (cm)", min_value=0)
massa = st.number_input("Massa (kg)", min_value=0)
dinamometria_md = st.number_input("Dinamometria MD", min_value=0)
dinamometria_mnd = st.number_input("Dinamometria MND", min_value=0)
cmj = st.number_input("CMJ (cm)", min_value=0)
sentar_alcancar = st.number_input("Sentar e Alcançar (cm)", min_value=0)
medicine_ball = st.number_input("Medicine Ball (m)", min_value=0)
corrida_20m = st.number_input("Corrida 20m (s)", min_value=0)
abdominal = st.number_input("Abdominal (qntd.)", min_value=0)
trinta_quinze = st.number_input("30/15 (km/h)", min_value=0)
quadrado = st.number_input("Quadrado (s)", min_value=0)

# Preparar os dados de entrada para fazer a previsão
new_data = pd.DataFrame({
    'nome': [nome],
    'Idade': [idade],
    'Estatura (cm)': [estatura],
    'Massa (kg)': [massa],
    'Dinamometria MD': [dinamometria_md],
    'Dinamometria MND': [dinamometria_mnd],
    'CMJ (cm)': [cmj],
    'Sentar e alcançar (cm)': [sentar_alcancar],
    'Medicine ball (m)': [medicine_ball],
    'Corrida 20 m (s)': [corrida_20m],
    'Abdominal (qntd.)': [abdominal],
    '30/15 (km/h)': [trinta_quinze],
    'Quadrado (s)': [quadrado]
})

# Fazer a previsão para os novos atletas
predicted_data = predict_super_athlete(new_data, scaler, kmeans, best_cluster)

# Exibir a previsão
st.write("Resultado da Previsão:")
st.write(predicted_data[['nome', 'is_super_athlete']])
