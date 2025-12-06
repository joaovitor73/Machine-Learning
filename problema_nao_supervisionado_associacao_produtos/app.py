import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import plotly.express as px
from pyvis.network import Network
import tempfile

st.set_page_config(page_title="Retail Rules Explorer", layout="wide")

st.title("🛒 Sistema de Recomendação de Varejo")
st.markdown("Descubra padrões de compra filtrando por **País** (ou global) e **Intervalo de Sazonalidade**.")

@st.cache_data
def load_data():
    from ucimlrepo import fetch_ucirepo
    retail = fetch_ucirepo(id=352)
    df = retail.data.original.copy()
    
    # Limpeza básica
    df = df.dropna(subset=['InvoiceNo'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    
    # Remover devoluções e preços errados
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Converter data e extrair mês
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month
    
    return df

try:
    df = load_data()
    st.success("Dados carregados com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()


st.sidebar.header("Filtros de Análise")


lista_paises = sorted(list(df['Country'].unique()))
lista_paises.insert(0, "Todos") 

pais_selecionado = st.sidebar.selectbox(
    "Escolha o País", 
    options=lista_paises, 
    index=0 
)

mes_inicio, mes_fim = st.sidebar.slider(
    "Intervalo de Meses", 
    min_value=1, 
    max_value=12, 
    value=(1, 12),
    help="Selecione o mês inicial e final para a análise."
)

min_sup = st.sidebar.slider("Suporte Mínimo", 0.01, 0.2, 0.02, help="Valores menores encontram mais regras, mas exige mais processamento.")

if st.sidebar.button("Gerar Regras"):
    
    # Filtro de País
    if pais_selecionado == "Todos":
        df_country = df
    else:
        df_country = df[df['Country'] == pais_selecionado]
        
    # Filtro de Meses (Intervalo)
    dados_filtro = df_country[
        (df_country['Month'] >= mes_inicio) & 
        (df_country['Month'] <= mes_fim)
    ]
    
    if len(dados_filtro) == 0:
        st.warning("Sem vendas registradas para este filtro.")
    else:
        st.info(f"Analisando {len(dados_filtro)} transações entre Mês {mes_inicio} e {mes_fim} em '{pais_selecionado}'...")
        
        # Cria a cesta (One-hot encoding simples via pivot)
        cesta = (dados_filtro.groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

        # Converter para booleano (Recomendado pelo mlxtend)
        cesta_encoded = (cesta >= 1)
    
        # Remover 'POSTAGE' se existir, pois geralmente distorce as regras
        if 'POSTAGE' in cesta_encoded.columns:
            print( "Dropping POSTAGE column")
            cesta_encoded.drop('POSTAGE', inplace=True, axis=1)

        # Apriori
        try:
            frequent_items = apriori(cesta_encoded, min_support=min_sup, use_colnames=True)
        except MemoryError:
            st.error("Memória insuficiente! O conjunto de dados 'Todos' é muito grande para este Suporte Mínimo. Tente aumentar o Suporte ou filtrar um país específico.")
            st.stop()
        
        if frequent_items.empty:
            st.warning("Nenhuma regra encontrada. Tente diminuir o Suporte Mínimo.")
        else:
            regras = association_rules(frequent_items, metric="lift", min_threshold=1)

            # Função auxiliar para strings
            def frozenset_to_str(x):
                if isinstance(x, (set, frozenset)):
                    return ", ".join(sorted(map(str, x)))
                return str(x)

            if not regras.empty:
                regras = regras.copy()
                if 'antecedents' in regras.columns:
                    regras['antecedents_str'] = regras['antecedents'].apply(frozenset_to_str)
                if 'consequents' in regras.columns:
                    regras['consequents_str'] = regras['consequents'].apply(frozenset_to_str)
                
                # --- 4. RESULTADOS NA TELA ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Top Regras (Maior Lift)")
                    st.caption("Produtos que aumentam muito a chance de compra um do outro.")
                    display_cols = ['antecedents_str', 'consequents_str', 'lift', 'confidence', 'support']
                    st.dataframe(
                        regras[display_cols]
                        .sort_values('lift', ascending=False)
                        .head(10)
                        .style.format({"lift": "{:.2f}", "confidence": "{:.2%}", "support": "{:.3f}"})
                    )

                with col2:
                    st.subheader("Gráfico de Dispersão")
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
                            "confidence": "Certeza (Confiança)"
                        },
                        title="Quanto mais alto e à direita, mais forte a regra",
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("🕸️ Rede de Conexões Interativa")
                st.caption("Visualizando as 20 conexões mais fortes (baseado em Lift).")
                
                # Limitamos a 20 ou 30 para o grafo não travar o navegador
                top_regras = regras.sort_values('lift', ascending=False).head(30)

                G_vis = nx.from_pandas_edgelist(
                    top_regras, 
                    source='antecedents_str',
                    target='consequents_str',
                    edge_attr='lift'
                )

                net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
                net.from_nx(G_vis)
                
                for node in net.nodes:
                    node['title'] = node['id'] 
                    node['color'] = '#FF6347' 
                    node['size'] = 20
                    node['font'] = {'size': 14}
                
                # Física ajustada para separar melhor os clusters
                net.repulsion(node_distance=150, spring_length=200)

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                        path = tmp_file.name
                        net.save_graph(path)
                        with open(path, 'r', encoding='utf-8') as f:
                            html_string = f.read()
                        components.html(html_string, height=550)
                except Exception as e:
                    st.error(f"Erro ao gerar grafo interativo: {e}")

            else:
                st.warning("Regras geradas, mas tabela vazia (verifique os parâmetros).")