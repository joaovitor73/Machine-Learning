import streamlit as st
import pandas as pd
# 'joblib' não é mais usado se estamos usando pickle
# import joblib 
import warnings
import pickle   

# Ignorar avisos futuros (opcional, mas limpa o output)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Empréstimos",
    page_icon="📊",
    layout="wide"
)

# --- Cache do Modelo ---
@st.cache_resource
def load_model():
    """Carrega o pipeline do modelo salvo (com pickle)."""
    
    # --- MUDANÇA 1: Corrigido o nome do arquivo ---
    # Para bater EXATAMENTE com o seu script de salvamento
    # (sem a pasta 'models/')
    filename = 'models/final_loan_model.pkl' 
    
    try:
        # Abre o arquivo em modo 'read binary' (rb)
        with open(filename, 'rb') as f:
            
            # --- MUDANÇA 2: Carregar o dicionário ---
            # O arquivo .pkl contém um dicionário, não o pipeline
            model_data = pickle.load(f)
        
        # --- MUDANÇA 3: Extrair o pipeline de dentro do dict ---
        # Acessamos a chave "model" que você definiu ao salvar
        pipeline = model_data["model"]
        
        return pipeline
        
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{filename}' não foi encontrado.")
        st.info("Por favor, execute o script de salvamento no mesmo diretório.")
        return None
    except KeyError:
        # Este erro acontece se o .pkl não tiver a chave "model"
        st.error(f"Erro: O arquivo '{filename}' foi encontrado, mas não contém a chave 'model'.")
        st.info("Verifique se seu script de salvamento está salvando um dict: {'model': ...}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None

# Carregar o modelo
# 'model' agora será o pipeline, extraído do dicionário
model = load_model()

# --- Título e Descrição ---
st.title("📊 Simulador de Aprovação de Empréstimos")
st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning (treinado com dados
históricos) para prever a probabilidade de aprovação de um pedido de empréstimo.

**Preencha os dados do solicitante na barra lateral à esquerda.**
""")

# --- (j) Formulário do Usuário (na barra lateral) ---
st.sidebar.header("📝 Dados do Solicitante")

# Mapeamentos (conforme os dados originais)
gender_options = ['Male', 'Female']
married_options = ['Yes', 'No']
# Conforme a etapa (f), o pipeline foi treinado com '3' em vez de '3+'
dependents_options = ['0', '1', '2', '3']
education_options = ['Graduate', 'Not Graduate']
self_employed_options = ['No', 'Yes']

# Criar os controles do formulário
gender = st.sidebar.radio("Sexo:", gender_options)
married = st.sidebar.radio("Casado:", married_options)
dependents = st.sidebar.selectbox("Dependentes:", dependents_options)
education = st.sidebar.selectbox("Educação (Nível Superior):", education_options)
self_employed = st.sidebar.radio("Autônomo:", self_employed_options)
applicantincome = st.sidebar.number_input("Renda do Solicitante (mensal):",
                                          min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Valor do Empréstimo (em milhares):",
                                      min_value=0, value=150)

# Botão para submeter
submit_button = st.sidebar.button("Analisar Empréstimo", type="primary")

# --- (k) Exibição dos Resultados ---
if submit_button and model is not None:

    # 1. Criar o DataFrame de entrada
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicantincome],
        'LoanAmount': [loan_amount]
    })

    st.subheader("Dados Fornecidos:")
    st.dataframe(input_data)

    try:
        # 2. Fazer a predição
        # Agora 'model' é o pipeline e isso vai funcionar
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)

        # 3. Interpretar os resultados
        result_status = prediction[0]
        confidence = proba[0][result_status]

        # 4. Exibir o resultado final
        st.write("---")
        st.subheader("Resultado da Análise:")

        if result_status == 1:
            st.success("🎉 **Status: APROVADO**")
            st.markdown(f"O modelo está **{confidence:.2%}** confiante de que este empréstimo deve ser **aprovado**.")
        else:
            st.error("❌ **Status: NEGADO**")
            st.markdown(f"O modelo está **{confidence:.2%}** confiante de que este empréstimo deve ser **negado**.")

        # Exibindo as probabilidades de ambas as classes (opcional)
        st.subheader("Probabilidades Detalhadas:")
        prob_df = pd.DataFrame({
            'Classe': ['Negado (0)', 'Aprovado (1)'],
            'Probabilidade': proba[0]
        })
        st.bar_chart(prob_df.set_index('Classe'))

    except Exception as e:
        st.error(f"Erro ao processar a predição: {e}")
        # --- MUDANÇA 4: Atualizada a mensagem de erro ---
        st.error(f"Verifique se o modelo '{filename}' está correto e compatível.")