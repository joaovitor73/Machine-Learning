import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from pyvis.network import Network
import tempfile

st.set_page_config(page_title="Retail Rules Explorer", layout="wide")

st.title("🛒 Sistema de Recomendação de Varejo")
st.markdown("Descubra padrões de compra baseados em **País** e **Sazonalidade**.")

@st.cache_data
def load_data():
    from ucimlrepo import fetch_ucirepo
    retail = fetch_ucirepo(id=352)
    df = retail.data.original.copy()
    df = df.dropna(subset=['InvoiceNo'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month
    return df

try:
    df = load_data()
    st.success("Dados carregados com sucesso!")
except:
    st.error("Erro ao carregar dados. Verifique a conexão.")

st.sidebar.header("Filtros de Análise")
paises = df['Country'].unique()
pais_selecionado = st.sidebar.selectbox("Escolha o País", options=paises, index=list(paises).index('France'))

mes_selecionado = st.sidebar.slider("Mês do Ano", 1, 12, 11)

min_sup = st.sidebar.slider("Suporte Mínimo", 0.01, 0.2, 0.05)

if st.sidebar.button("Gerar Regras"):
    dados_filtro = df[(df['Country'] == pais_selecionado) & (df['Month'] == mes_selecionado)]
    
    if len(dados_filtro) == 0:
        st.warning("Sem vendas registradas para este filtro.")
    else:
        st.info(f"Analisando {len(dados_filtro)} transações...")
        
        # Cria a cesta
        cesta = (dados_filtro.groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

        # Use boolean dtype (recommended by mlxtend) instead of numeric ints
        cesta_encoded = (cesta >= 1)
        if 'POSTAGE' in cesta_encoded.columns:
            cesta_encoded.drop('POSTAGE', inplace=True, axis=1)

        # Apriori
        frequent_items = apriori(cesta_encoded, min_support=min_sup, use_colnames=True)
        
        if frequent_items.empty:
            st.warning("Nenhuma regra encontrada. Tente diminuir o Suporte Mínimo.")
        else:
            regras = association_rules(frequent_items, metric="lift", min_threshold=1)

            # Stringify frozenset columns so Streamlit / pyarrow can serialize them
            # and networkx can use labels. Create display columns.
            def frozenset_to_str(x):
                if isinstance(x, (set, frozenset)):
                    return ", ".join(sorted(map(str, x)))
                return str(x)

            regras = regras.copy()
            if 'antecedents' in regras.columns:
                regras['antecedents_str'] = regras['antecedents'].apply(frozenset_to_str)
            if 'consequents' in regras.columns:
                regras['consequents_str'] = regras['consequents'].apply(frozenset_to_str)
            
            # --- 4. RESULTADOS NA TELA ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Top Regras (Maior Lift)")
                st.dataframe(regras[['antecedents_str', 'consequents_str', 'lift', 'confidence']].sort_values('lift', ascending=False).head(10))

            with col2:

                fig = px.scatter(
                    regras,
                    x="support",
                    y="lift",
                    size="confidence",
                    color="lift",
                    hover_data=["antecedents_str", "consequents_str"],
                    labels={
                        "support": "Popularidade (Suporte)",
                        "lift": "Força da Associação (Lift)",
                        "confidence": "Certeza (Confiança)",
                        "lift": "Lift"
                    },
                    title="Quanto mais alto e à direita, melhor a regra!",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Rede de Conexões Interativa")
                st.caption("Arraste os nós para explorar as conexões!")

                # OPÇÃO 2: REDE INTERATIVA (PyVis)
                # Cria um grafo onde o usuário pode mexer
                
                # 1. Criar o grafo NetworkX (igual você já fazia, mas limpo)
                G_vis = nx.from_pandas_edgelist(
                    regras.head(20), # Limitado a 20 para não virar uma "bola de lã"
                    source='antecedents_str',
                    target='consequents_str',
                    edge_attr='lift'
                )

                # 2. Configurar PyVis
                net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black")
                net.from_nx(G_vis)
                
                # 3. Personalizar a física e cores dos nós para ficar bonito
                for node in net.nodes:
                    node['title'] = node['id'] # Tooltip ao passar o mouse
                    node['color'] = '#FF6347' # Cor tomate para os produtos
                    node['size'] = 20
                
                # Ajustar física (para os nós não ficarem um em cima do outro)
                net.repulsion(node_distance=200, spring_length=200)

                # 4. Salvar e exibir no Streamlit (Truque do arquivo temporário)
                try:
                    # Cria um arquivo temporário para salvar o HTML do grafo
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                        path = tmp_file.name
                        net.save_graph(path)
                        
                        # Ler o arquivo e exibir
                        with open(path, 'r', encoding='utf-8') as f:
                            html_string = f.read()
                        
                        components.html(html_string, height=450)
                except Exception as e:
                    st.error(f"Erro ao gerar grafo interativo: {e}")
                    st.info("Tente usar o gráfico de bolhas acima.")