"""
Página Streamlit com três painéis.

Painel 1: análise descritiva da população de admissões do Novo CAGED 2025,
lida de saida/populacao.parquet (gerado por ScriptParaBaixar.py).
Painel 2: a mesma análise, feita sobre a amostra (saida/amostra.parquet).
Painel 3: inferência (IC e testes de hipóteses) com a amostra, comparada
com os parâmetros calculados diretamente na população.
As agregações são feitas com DuckDB direto no parquet, sem carregar
as ~26 milhões de linhas na memória.

Uso:
    pip install streamlit duckdb pandas plotly scipy
    python -m streamlit run painel.py
"""

import math
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

PASTA_SCRIPT = Path(__file__).resolve().parent
ARQ_POP = PASTA_SCRIPT / "saida" / "populacao.parquet"
if not ARQ_POP.exists():
    # No deploy (Streamlit Cloud) só vai a versão enxuta: apenas as colunas usadas
    # pelo painel, com compressão ZSTD, para caber no limite de 100 MB do GitHub.
    ARQ_POP = PASTA_SCRIPT / "saida" / "populacao_deploy.parquet"
ARQ_AMOSTRA = PASTA_SCRIPT / "saida" / "amostra.parquet"

COR = "#2a78d6"  # cor única das séries (uma série por gráfico)


def sem_latex(texto: str) -> str:
    """O Streamlit trata texto entre dois "$" como fórmula LaTeX; escapa o cifrão (R$)."""
    return texto.replace("$", r"\$")


def markdown(texto: str, **kwargs) -> None:
    st.markdown(sem_latex(texto), **kwargs)


def legenda(texto: str, **kwargs) -> None:
    st.caption(sem_latex(texto), **kwargs)

# ----------------------------------------------------------------------
# Painel 3 - configuração da inferência
# alternativa: "greater" (H1: >), "less" (H1: <) ou "two-sided" (H1: ≠)
# ----------------------------------------------------------------------
ALFA = 0.05
HIP_PROPORCAO = dict(p0=0.50, alternativa="less")
HIP_MEDIA = dict(mu0=3652.00, alternativa="less",
                 fonte_mu0="salário médio do brasileiro divulgado pelo IBGE")

# ----------------------------------------------------------------------
# Dicionários de rótulos (Novo CAGED / IBGE / CNAE 2.0)
# ----------------------------------------------------------------------
SEXOS = {1: "Homem", 3: "Mulher"}
REGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
UFS = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
# Ordem natural dos níveis de instrução (variável ordinal)
GRAU_INSTRUCAO = {
    1: "Analfabeto", 2: "Até 5º ano incompleto", 3: "5º ano completo",
    4: "6º ao 9º ano incompleto", 5: "Fundamental completo", 6: "Médio incompleto",
    7: "Médio completo", 8: "Superior incompleto", 9: "Superior completo",
    80: "Pós-graduação completa", 10: "Mestrado", 11: "Doutorado", 99: "Não identificado",
}
GRAU_SUPERIOR = (9, 10, 11, 80)
RACAS = {1: "Branca", 2: "Preta", 3: "Parda", 4: "Amarela", 5: "Indígena",
         6: "Não informada", 9: "Não identificada"}
SECOES = {
    "A": "Agropecuária", "B": "Indústrias extrativas", "C": "Indústria de transformação",
    "D": "Eletricidade e gás", "E": "Água, esgoto e resíduos", "F": "Construção",
    "G": "Comércio e reparação de veículos", "H": "Transporte e armazenagem",
    "I": "Alojamento e alimentação", "J": "Informação e comunicação",
    "K": "Atividades financeiras", "L": "Atividades imobiliárias",
    "M": "Ativ. profissionais e científicas", "N": "Ativ. administrativas e serv. complementares",
    "O": "Administração pública", "P": "Educação", "Q": "Saúde e serviços sociais",
    "R": "Artes, cultura e recreação", "S": "Outras atividades de serviços",
    "T": "Serviços domésticos", "U": "Organismos internacionais", "Z": "Não identificado",
}

QUANTITATIVAS = {"salario": "Salário (R$)", "idade": "Idade (anos)", "horas": "Horas semanais"}
QUALITATIVAS = {
    # coluna: (rótulo, dicionário, ordenar pela ordem do dicionário?)
    "sexo": ("Sexo", SEXOS, False),
    "regiao": ("Região", REGIOES, False),
    "uf": ("UF", UFS, False),
    "grau_instrucao": ("Escolaridade", GRAU_INSTRUCAO, True),
    "secao": ("Setor (seção CNAE)", SECOES, False),
    "raca_cor": ("Raça/cor", RACAS, False),
    "trab_parcial": ("Trabalho parcial", {0: "Não", 1: "Sim", 9: "Não identificado"}, False),
}
# Variáveis qualitativas exibidas em abas no Painel 1
# (sexo, raça/cor e trabalho parcial ficam de fora; sexo segue usado no salário por grupo)
ABAS_QUALITATIVAS = ["regiao", "uf", "grau_instrucao", "secao"]
# Variáveis exibidas em gráfico dividido por sexo (homens à esquerda, mulheres à direita)
POR_SEXO = {"grau_instrucao", "secao"}
COR_HOMEM = "#2a78d6"   # azul
COR_MULHER = "#e87ba4"  # rosa
# Referência da população no painel da amostra: traço fino cinza (legível nos temas claro e escuro)
COR_REF = "rgba(128, 128, 128, 0.75)"
NOME_REF = "População (referência)"


# ----------------------------------------------------------------------
# Formatação pt-BR
# ----------------------------------------------------------------------
def br(valor: float, casas: int = 2) -> str:
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(valor: float, casas: int = 1) -> str:
    return br(valor * 100, casas) + "%"


# ----------------------------------------------------------------------
# Consultas (cache invalida quando o parquet é regerado)
# ----------------------------------------------------------------------
def consulta(fonte: tuple, sql: str) -> pd.DataFrame:
    """fonte = (caminho do parquet, data de modificação); {pop} vira o caminho."""
    with duckdb.connect(config={"memory_limit": "1GB", "threads": 2}) as con:
        return con.execute(sql.replace("{pop}", f"'{fonte[0]}'")).df()


@st.cache_data(show_spinner="Calculando indicadores...")
def indicadores(fonte: tuple) -> dict:
    superior = ", ".join(map(str, GRAU_SUPERIOR))
    df = consulta(fonte, f"""
        SELECT count(*)                                    AS total,
               avg(salario)                                AS sal_medio,
               median(salario)                             AS sal_mediano,
               avg(idade)                                  AS idade_media,
               avg(horas)                                  AS horas_media,
               avg(CASE WHEN sexo = 3 THEN 1 ELSE 0 END)   AS prop_mulheres,
               avg(CASE WHEN grau_instrucao IN ({superior}) THEN 1 ELSE 0 END) AS prop_superior,
               count(DISTINCT municipio)                   AS municipios
        FROM {{pop}}
    """)
    return df.iloc[0].to_dict()


@st.cache_data(show_spinner="Calculando medidas descritivas...")
def medidas_quantitativas(fonte: tuple) -> pd.DataFrame:
    linhas = []
    for col, rotulo in QUANTITATIVAS.items():
        # Todos os quantis numa única chamada (forma de lista): o DuckDB guarda uma só
        # cópia dos valores, em vez de uma por quantil - cabe na memória do Streamlit Cloud.
        r = consulta(fonte, f"""
            SELECT n, media, dp, minimo, max_ AS maximo, assimetria, curtose,
                   qs[1] AS p1, qs[2] AS q1, qs[3] AS mediana, qs[4] AS q3, qs[5] AS p99
            FROM (
                SELECT count({col}) AS n, avg({col}) AS media,
                       stddev_samp({col}) AS dp, min({col}) AS minimo, max({col}) AS max_,
                       quantile_cont({col}, [0.01, 0.25, 0.5, 0.75, 0.99]) AS qs,
                       skewness({col}) AS assimetria, kurtosis({col}) AS curtose
                FROM {{pop}}
            )
        """).iloc[0]
        linhas.append({
            "Variável": rotulo, "Média": r.media, "Mediana": r.mediana,
            "Desvio-padrão": r.dp, "CV (%)": r.dp / r.media * 100,
            "Mínimo": r.minimo, "P1": r.p1, "Q1": r.q1, "Q3": r.q3,
            "AIQ (Q3−Q1)": r.q3 - r.q1, "P99": r.p99, "Máximo": r.maximo,
            "Assimetria": r.assimetria, "Curtose (excesso)": r.curtose,
        })
    return pd.DataFrame(linhas).set_index("Variável")


@st.cache_data(show_spinner="Montando distribuição...")
def distribuicao(fonte: tuple, col: str, largura: float, limite_sup: float) -> pd.DataFrame:
    return consulta(fonte, f"""
        SELECT floor({col} / {largura}) * {largura} AS inicio, count(*) AS n
        FROM {{pop}}
        WHERE {col} <= {limite_sup}
        GROUP BY 1 ORDER BY 1
    """)


@st.cache_data(show_spinner="Calculando frequências...")
def frequencias(fonte: tuple, col: str) -> pd.DataFrame:
    return consulta(fonte, f"SELECT {col} AS codigo, count(*) AS n FROM {{pop}} GROUP BY 1")


@st.cache_data(show_spinner="Calculando frequências por sexo...")
def frequencias_por_sexo(fonte: tuple, col: str) -> pd.DataFrame:
    """Contagem por categoria e sexo, em colunas: codigo, n_homens, n_mulheres."""
    df = consulta(fonte, f"""
        SELECT {col} AS codigo,
               count(*) FILTER (WHERE sexo = 1) AS n_homens,
               count(*) FILTER (WHERE sexo = 3) AS n_mulheres
        FROM {{pop}} GROUP BY 1
    """)
    return df


@st.cache_data(show_spinner="Calculando quartis por grupo...")
def quartis_por_grupo(fonte: tuple, col_valor: str, col_grupo: str) -> pd.DataFrame:
    return consulta(fonte, f"""
        SELECT codigo, n, media, minimo, maximo, qs[1] AS q1, qs[2] AS mediana, qs[3] AS q3
        FROM (
            SELECT {col_grupo} AS codigo, count(*) AS n, avg({col_valor}) AS media,
                   min({col_valor}) AS minimo, max({col_valor}) AS maximo,
                   quantile_cont({col_valor}, [0.25, 0.5, 0.75]) AS qs
            FROM {{pop}} GROUP BY 1
        )
    """)


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def layout_padrao(fig: go.Figure, titulo_x: str, titulo_y: str, altura: int = 380) -> go.Figure:
    fig.update_layout(
        height=altura, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        xaxis_title=titulo_x, yaxis_title=titulo_y, bargap=0.05,
        separators=",.",
    )
    fig.update_yaxes(showgrid=True, gridwidth=1)
    fig.update_xaxes(showgrid=False)
    return fig


def legenda_referencia(fig: go.Figure) -> go.Figure:
    """Mostra a legenda (amostra × população) no topo do gráfico."""
    fig.update_layout(
        showlegend=True, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5),
    )
    return fig


def histograma(dist: pd.DataFrame, largura: float, rotulo: str, unidade_fmt,
               ref: pd.DataFrame | None = None) -> go.Figure:
    """ref (opcional): colunas inicio, esperado, prop - população reescalada ao tamanho da amostra."""
    fim = dist["inicio"] + largura
    faixas = [f"{unidade_fmt(a)} a {unidade_fmt(b)}" for a, b in zip(dist["inicio"], fim)]
    fig = go.Figure(go.Bar(
        name="Amostra", x=dist["inicio"] + largura / 2, y=dist["n"], width=largura * 0.95,
        marker_color=COR, customdata=faixas,
        hovertemplate="%{customdata}<br>%{y:,.0f} admissões<extra></extra>",
    ))
    fig = layout_padrao(fig, rotulo, "Admissões")
    if ref is not None:
        fim_ref = ref["inicio"] + largura
        fig.add_trace(go.Scatter(
            name=NOME_REF, x=ref["inicio"] + largura / 2, y=ref["esperado"],
            mode="lines", line=dict(color=COR_REF, width=1.5, shape="hvh"),
            customdata=list(zip([f"{unidade_fmt(a)} a {unidade_fmt(b)}"
                                 for a, b in zip(ref["inicio"], fim_ref)],
                                [pct(p, 2) for p in ref["prop"]])),
            hovertemplate="%{customdata[0]}<br>População: %{customdata[1]}<extra></extra>",
        ))
        legenda_referencia(fig)
    return fig


def boxplot(q: pd.DataFrame, rotulos: list[str], titulo_x: str, titulo_y: str) -> go.Figure:
    """Boxplot a partir de quartis pré-calculados; bigodes = 1,5×AIQ limitados a mín/máx."""
    aiq = q["q3"] - q["q1"]
    inferior = (q["q1"] - 1.5 * aiq).clip(lower=q["minimo"])
    superior = (q["q3"] + 1.5 * aiq).clip(upper=q["maximo"])
    fig = go.Figure(go.Box(
        x=rotulos, q1=q["q1"], median=q["mediana"], q3=q["q3"], mean=q["media"],
        lowerfence=inferior, upperfence=superior,
        marker_color=COR, fillcolor="rgba(42,120,214,0.25)", line_width=2,
        boxmean=True,
    ))
    return layout_padrao(fig, titulo_x, titulo_y)


def barras(freq: pd.DataFrame, horizontal: bool, ref_prop: pd.Series | None = None) -> go.Figure:
    """ref_prop (opcional): proporção da população em cada categoria, alinhada com freq.
    É desenhada como um traço fino no valor esperado (proporção × total da amostra)."""
    texto = [pct(p) for p in freq["Proporção"]]
    hover = "%{customdata[0]}<br>%{customdata[1]:,.0f} admissões (%{customdata[2]})<extra></extra>"
    dados = list(zip(freq["Categoria"], freq["Frequência"], texto))
    if horizontal:
        fig = go.Figure(go.Bar(
            name="Amostra", y=freq["Categoria"], x=freq["Frequência"], orientation="h",
            marker_color=COR, text=texto, textposition="outside", cliponaxis=False,
            customdata=dados, hovertemplate=hover,
        ))
        fig.update_yaxes(autorange="reversed")
        altura = max(320, 26 * len(freq) + 60)
        fig = layout_padrao(fig, "Admissões", "", altura)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)
    else:
        fig = go.Figure(go.Bar(
            name="Amostra", x=freq["Categoria"], y=freq["Frequência"], marker_color=COR,
            text=texto, textposition="outside", cliponaxis=False,
            customdata=dados, hovertemplate=hover,
        ))
        fig = layout_padrao(fig, "", "Admissões")

    if ref_prop is not None:
        esperado = ref_prop.to_numpy() * freq["Frequência"].sum()
        eixo_cat, eixo_val = ("y", "x") if horizontal else ("x", "y")
        fig.add_trace(go.Scatter(
            name=NOME_REF, mode="markers", cliponaxis=False,
            **{eixo_cat: freq["Categoria"], eixo_val: esperado},
            marker=dict(symbol="line-ns" if horizontal else "line-ew", size=22 if horizontal else 40,
                        line=dict(color=COR_REF, width=2)),
            customdata=list(zip(freq["Categoria"], [pct(p) for p in ref_prop])),
            hovertemplate="%{customdata[0]}<br>População: %{customdata[1]}<extra></extra>",
        ))
        legenda_referencia(fig)
    return fig


def piramide_sexo(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> go.Figure:
    """Barras divididas ao meio: % dos homens à esquerda (azul), % das mulheres à direita (rosa).
    Cada percentual é calculado dentro do próprio sexo (soma 100% de cada lado).
    ref (opcional): mesmas colunas para a população, alinhada com df (traço fino cinza)."""
    maximo = max(df["% dos homens"].max(), df["% das mulheres"].max())
    if ref is not None:
        maximo = max(maximo, ref["% dos homens"].max(), ref["% das mulheres"].max())
    passo = next(p for p in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5) if maximo / p <= 4)
    marcas = [i * passo for i in range(-int(maximo / passo) - 1, int(maximo / passo) + 2)]
    hover = "%{customdata[0]}<br>%{customdata[1]}: %{customdata[2]} (%{customdata[3]:,.0f})<extra></extra>"

    fig = go.Figure()
    for nome, col_pct, col_n, cor, sinal in [
        ("Homens", "% dos homens", "n_homens", COR_HOMEM, -1),
        ("Mulheres", "% das mulheres", "n_mulheres", COR_MULHER, 1),
    ]:
        texto = [pct(v) for v in df[col_pct]]
        fig.add_trace(go.Bar(
            name=nome, y=df["Categoria"], x=sinal * df[col_pct], orientation="h",
            marker_color=cor, text=texto, textposition="outside", cliponaxis=False,
            customdata=list(zip(df["Categoria"], [nome] * len(df), texto, df[col_n])),
            hovertemplate=hover,
        ))

    if ref is not None:
        for nome, col_pct, sinal in [("Homens", "% dos homens", -1),
                                     ("Mulheres", "% das mulheres", 1)]:
            fig.add_trace(go.Scatter(
                name=NOME_REF, legendgroup="ref", showlegend=(sinal == -1),
                y=ref["Categoria"], x=sinal * ref[col_pct], mode="markers", cliponaxis=False,
                marker=dict(symbol="line-ns", size=22, line=dict(color=COR_REF, width=2)),
                customdata=list(zip(ref["Categoria"], [pct(v) for v in ref[col_pct]])),
                hovertemplate=f"%{{customdata[0]}}<br>População ({nome.lower()}): "
                              "%{customdata[1]}<extra></extra>",
            ))

    altura = max(360, 30 * len(df) + 90)
    fig = layout_padrao(fig, "Percentual dentro de cada sexo", "", altura)
    fig.update_layout(
        barmode="relative", bargap=0.2, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(
        tickvals=marcas, ticktext=[pct(abs(m), 0) for m in marcas],
        range=[-maximo * 1.3, maximo * 1.3], showgrid=True, zeroline=True, zerolinewidth=2,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def ref_histograma(fonte_ref: tuple | None, col: str, largura: float, limite_sup: float,
                   n_amostra: int) -> pd.DataFrame | None:
    """Distribuição da população nas mesmas faixas, reescalada ao tamanho da amostra."""
    if fonte_ref is None:
        return None
    ref = distribuicao(fonte_ref, col, largura, limite_sup)
    total_pop = indicadores(fonte_ref)["total"]
    ref["prop"] = ref["n"] / total_pop
    ref["esperado"] = ref["prop"] * n_amostra
    return ref


def tabela_frequencias(cod: pd.DataFrame, dicionario: dict, ordem_natural: bool) -> pd.DataFrame:
    df = cod.copy()
    df["Categoria"] = df["codigo"].map(dicionario).fillna(df["codigo"].astype(str))
    if ordem_natural:
        ordem = {k: i for i, k in enumerate(dicionario)}
        df = df.sort_values("codigo", key=lambda s: s.map(ordem).fillna(len(ordem)))
    else:
        df = df.sort_values("n", ascending=False)
    df = df.rename(columns={"n": "Frequência"})
    df["Proporção"] = df["Frequência"] / df["Frequência"].sum()
    df["Proporção acumulada"] = df["Proporção"].cumsum()
    return df[["Categoria", "Frequência", "Proporção", "Proporção acumulada"]].reset_index(drop=True)


# ----------------------------------------------------------------------
# Painel 1 - análise descritiva da população
# ----------------------------------------------------------------------
def painel_descritivo(arquivo: Path, titulo: str, descricao: str,
                      referencia: Path | None = None) -> None:
    """referencia (opcional): parquet da população, desenhado como traço fino nos
    gráficos de barras quando o usuário ativa a opção."""
    if not arquivo.exists():
        st.error(f"Arquivo não encontrado: {arquivo}. Rode ScriptParaBaixar.py antes.")
        return
    fonte = (arquivo.as_posix(), arquivo.stat().st_mtime)

    st.header(titulo)
    legenda(descricao)

    fonte_ref = None
    if referencia is not None and referencia.exists():
        if st.toggle("Mostrar a população como referência nos gráficos de barras",
                     value=True, key=f"ref_{arquivo.stem}"):
            fonte_ref = (referencia.as_posix(), referencia.stat().st_mtime)
            legenda("O traço cinza marca onde a barra estaria se a amostra tivesse exatamente "
                       "as proporções da população (proporção da população × tamanho da amostra).")

    # --- Indicadores -------------------------------------------------
    ind = indicadores(fonte)
    c = st.columns(4)
    c[0].metric("Total de admissões", br(ind["total"], 0))
    c[1].metric("Salário médio", "R$ " + br(ind["sal_medio"]))
    c[2].metric("Salário mediano", "R$ " + br(ind["sal_mediano"]))
    c[3].metric("Idade média", br(ind["idade_media"], 1) + " anos")
    c = st.columns(4)
    c[0].metric("Mulheres", pct(ind["prop_mulheres"]))
    c[1].metric("Com ensino superior", pct(ind["prop_superior"]))
    c[2].metric("Jornada média", br(ind["horas_media"], 1) + " h/semana")

    # --- Quantitativas -----------------------------------------------
    st.subheader("Variáveis quantitativas")
    medidas = medidas_quantitativas(fonte)
    st.dataframe(
        medidas.style.format(lambda v: br(v, 2)),
        width="stretch",
    )

    aba_sal, aba_idade, aba_horas = st.tabs(["Salário", "Idade", "Horas semanais"])

    with aba_sal:
        p99 = medidas.loc["Salário (R$)", "P99"]
        dist = distribuicao(fonte, "salario", 100, p99)
        ref = ref_histograma(fonte_ref, "salario", 100, p99, ind["total"])
        col1, col2 = st.columns([3, 2])
        with col1:
            markdown("**Distribuição do salário** (faixas de R$ 100, até o P99)")
            st.plotly_chart(histograma(dist, 100, "Salário (R$)", lambda v: "R$ " + br(v, 0), ref),
                            width="stretch")
        with col2:
            markdown("**Boxplot do salário**")
            q = quartis_por_grupo(fonte, "salario", "1")
            st.plotly_chart(boxplot(q, ["População"], "", "Salário (R$)"), width="stretch")
        legenda(f"O histograma omite o 1% de salários acima de R$ {br(p99, 0)} para não achatar "
                   "a distribuição, que tem forte assimetria à direita. "
                   "No boxplot, a linha tracejada é a média e os bigodes vão até 1,5×AIQ.")

        markdown("**Salário por grupo**")
        grupos = {"sexo": "Sexo", "regiao": "Região", "grau_instrucao": "Escolaridade"}
        for aba, (col, rotulo) in zip(st.tabs(list(grupos.values())), grupos.items()):
            with aba:
                _, dicionario, ordem_natural = QUALITATIVAS[col]
                q = quartis_por_grupo(fonte, "salario", col)
                if ordem_natural:
                    ordem = {k: i for i, k in enumerate(dicionario)}
                    q = q.sort_values("codigo", key=lambda s: s.map(ordem).fillna(len(ordem)))
                else:
                    q = q.sort_values("mediana", ascending=False)
                rotulos = q["codigo"].map(dicionario).fillna(q["codigo"].astype(str)).tolist()
                st.plotly_chart(boxplot(q, rotulos, rotulo, "Salário (R$)"),
                                width="stretch")

    with aba_idade:
        dist = distribuicao(fonte, "idade", 1, 200)
        ref = ref_histograma(fonte_ref, "idade", 1, 200, ind["total"])
        col1, col2 = st.columns([3, 2])
        with col1:
            markdown("**Distribuição da idade** (por ano de idade)")
            st.plotly_chart(histograma(dist, 1, "Idade (anos)", lambda v: br(v, 0), ref),
                            width="stretch")
        with col2:
            markdown("**Boxplot da idade**")
            q = quartis_por_grupo(fonte, "idade", "1")
            st.plotly_chart(boxplot(q, ["População"], "", "Idade (anos)"), width="stretch")

    with aba_horas:
        freq = frequencias(fonte, "horas").sort_values("n", ascending=False)
        total = freq["n"].sum()
        top = freq.head(10).copy()
        outros = total - top["n"].sum()
        top["Categoria"] = top["codigo"].map(lambda h: br(h, 0) + " h")
        tabela = pd.concat([top[["Categoria", "n"]],
                            pd.DataFrame({"Categoria": ["Outros valores"], "n": [outros]})])
        tabela = tabela.rename(columns={"n": "Frequência"})
        tabela["Proporção"] = tabela["Frequência"] / total
        ref_prop = None
        if fonte_ref is not None:
            pop = frequencias(fonte_ref, "horas")
            prop_pop = pop.set_index("codigo")["n"] / pop["n"].sum()
            ref_top = top["codigo"].map(prop_pop).fillna(0)
            ref_prop = pd.concat([ref_top, pd.Series([1 - ref_top.sum()])], ignore_index=True)
        markdown("**Jornadas mais frequentes** (10 valores mais comuns)")
        st.plotly_chart(barras(tabela, True, ref_prop), width="stretch")
        legenda("Como as horas se concentram em poucos valores (sobretudo 44 h), "
                   "um gráfico de barras dos valores mais comuns é mais informativo que um histograma.")

    # --- Qualitativas ------------------------------------------------
    st.subheader("Variáveis qualitativas")
    abas = st.tabs([QUALITATIVAS[col][0] for col in ABAS_QUALITATIVAS])
    for aba, col in zip(abas, ABAS_QUALITATIVAS):
        rotulo, dicionario, ordem_natural = QUALITATIVAS[col]
        with aba:
            freq = tabela_frequencias(frequencias(fonte, col), dicionario, ordem_natural)
            formatos = {
                "Frequência": lambda v: br(v, 0),
                "Proporção": lambda v: pct(v, 2),
                "Proporção acumulada": lambda v: pct(v, 2),
            }
            col1, col2 = st.columns([3, 2])

            if col in POR_SEXO:
                sexo = frequencias_por_sexo(fonte, col)
                sexo["Categoria"] = sexo["codigo"].map(dicionario).fillna(sexo["codigo"].astype(str))
                sexo["% dos homens"] = sexo["n_homens"] / sexo["n_homens"].sum()
                sexo["% das mulheres"] = sexo["n_mulheres"] / sexo["n_mulheres"].sum()
                # mesma ordem de categorias da tabela geral
                sexo = freq[["Categoria"]].merge(sexo, on="Categoria", how="left")
                freq = freq.merge(sexo[["Categoria", "% dos homens", "% das mulheres"]],
                                  on="Categoria")
                formatos |= {"% dos homens": lambda v: pct(v, 2),
                             "% das mulheres": lambda v: pct(v, 2)}
                ref = None
                if fonte_ref is not None:
                    pop = frequencias_por_sexo(fonte_ref, col)
                    pop["Categoria"] = pop["codigo"].map(dicionario).fillna(pop["codigo"].astype(str))
                    pop["% dos homens"] = pop["n_homens"] / pop["n_homens"].sum()
                    pop["% das mulheres"] = pop["n_mulheres"] / pop["n_mulheres"].sum()
                    ref = sexo[["Categoria"]].merge(pop, on="Categoria", how="left").fillna(0)
                with col1:
                    st.plotly_chart(piramide_sexo(sexo, ref), width="stretch")
                legenda("Cada lado soma 100%: o percentual dos homens é calculado sobre o total "
                           "de homens admitidos, e o das mulheres sobre o total de mulheres.")
            else:
                ref_prop = None
                if fonte_ref is not None:
                    pop = tabela_frequencias(frequencias(fonte_ref, col), dicionario, ordem_natural)
                    ref_prop = freq["Categoria"].map(pop.set_index("Categoria")["Proporção"]).fillna(0)
                with col1:
                    st.plotly_chart(barras(freq, len(freq) > 6, ref_prop), width="stretch")

            with col2:
                st.dataframe(freq.style.format(formatos), hide_index=True, width="stretch")
            if ordem_natural:
                legenda(f"{rotulo} é ordinal: as categorias seguem a ordem dos níveis, "
                           "não a frequência.")


# ----------------------------------------------------------------------
# Painel 3 - inferência (amostra) comparada com a população
# ----------------------------------------------------------------------
SIMBOLO_H1 = {"greater": ">", "less": "<", "two-sided": "≠"}
SIMBOLO_H0 = {"greater": "≤", "less": "≥", "two-sided": "="}
SENTIDO = {"greater": "mais do que", "less": "menos do que", "two-sided": "algo diferente de"}


@st.cache_data(show_spinner="Lendo a amostra...")
def dados_amostra(fonte: tuple) -> pd.DataFrame:
    return pd.read_parquet(fonte[0], columns=["sexo", "salario"])


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "< 0,0001"
    if p > 0.9999:
        return "> 0,9999"
    return br(p, 4)


def dif_pp(v: float) -> str:
    """Diferença entre proporções, em pontos percentuais."""
    return br(v * 100, 2) + " p.p."


def dif_real(v: float) -> str:
    return ("−" if v < 0 else "") + "R$ " + br(abs(v), 2)


def teste(estatistica: float, alternativa: str, dist) -> tuple[float, str, float]:
    """Valor-p, região de rejeição (texto) e valor crítico para a distribuição dada."""
    if alternativa == "greater":
        critico = dist.ppf(1 - ALFA)
        return dist.sf(estatistica), f"> {br(critico, 3)}", critico
    if alternativa == "less":
        critico = dist.ppf(ALFA)
        return dist.cdf(estatistica), f"< {br(critico, 3)}", critico
    critico = dist.ppf(1 - ALFA / 2)
    return 2 * dist.sf(abs(estatistica)), f"|·| > {br(critico, 3)}", critico


def inferencia_proporcao(x: pd.Series, p0: float, alternativa: str, p_pop: float) -> dict:
    n = len(x)
    p_hat = x.mean()
    z_ic = stats.norm.ppf(1 - ALFA / 2)
    ep = math.sqrt(p_hat * (1 - p_hat) / n)              # IC de Wald: EP com p-chapéu
    z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)      # teste: EP sob H0, com p0
    p_valor, regiao, _ = teste(z, alternativa, stats.norm)
    return dict(n=n, estimativa=p_hat, ep=ep, margem=z_ic * ep,
                ic_inf=p_hat - z_ic * ep, ic_sup=p_hat + z_ic * ep,
                estatistica=z, nome_estat="z", regiao=regiao, p_valor=p_valor,
                rejeita=p_valor < ALFA, valor0=p0, alternativa=alternativa, pop=p_pop)


def inferencia_media(x: pd.Series, mu0: float, alternativa: str, mu_pop: float) -> dict:
    n = len(x)
    media, s = x.mean(), x.std(ddof=1)
    dist = stats.t(df=n - 1)
    t_ic = dist.ppf(1 - ALFA / 2)
    ep = s / math.sqrt(n)
    t = (media - mu0) / ep
    p_valor, regiao, _ = teste(t, alternativa, dist)
    return dict(n=n, estimativa=media, dp=s, ep=ep, margem=t_ic * ep,
                ic_inf=media - t_ic * ep, ic_sup=media + t_ic * ep,
                estatistica=t, nome_estat="t", regiao=regiao, p_valor=p_valor,
                rejeita=p_valor < ALFA, valor0=mu0, alternativa=alternativa, pop=mu_pop,
                assimetria=stats.skew(x))


def h1_verdadeira_na_populacao(r: dict) -> bool:
    if r["alternativa"] == "greater":
        return r["pop"] > r["valor0"]
    if r["alternativa"] == "less":
        return r["pop"] < r["valor0"]
    return r["pop"] != r["valor0"]


def grafico_ic(r: dict, fmt, rotulo_h0: str) -> go.Figure:
    """IC 95% da amostra (barra), estimativa pontual (ponto) e parâmetro da população (traço)."""
    y = ["Amostra"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Estimativa e IC 95% (amostra)", x=[r["estimativa"]], y=y, mode="markers",
        marker=dict(color=COR, size=12),
        error_x=dict(type="data", symmetric=False, array=[r["ic_sup"] - r["estimativa"]],
                     arrayminus=[r["estimativa"] - r["ic_inf"]], thickness=3, width=12, color=COR),
        hovertemplate=(f"Estimativa: {fmt(r['estimativa'])}<br>"
                       f"IC 95%: [{fmt(r['ic_inf'])}; {fmt(r['ic_sup'])}]<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        name="Parâmetro da população", x=[r["pop"]], y=y, mode="markers",
        marker=dict(symbol="line-ns", size=34, line=dict(color=COR_REF, width=3)),
        hovertemplate=f"População: {fmt(r['pop'])}<extra></extra>",
    ))
    lo, hi = min(r["ic_inf"], r["pop"]), max(r["ic_sup"], r["pop"])
    folga = (hi - lo) * 0.6
    faixa = [lo - folga, hi + folga]
    if faixa[0] <= r["valor0"] <= faixa[1]:
        fig.add_vline(x=r["valor0"], line_dash="dash", line_color=COR_REF,
                      annotation_text=rotulo_h0, annotation_position="top")
    else:
        lado_direito = r["valor0"] > faixa[1]
        fig.add_annotation(
            x=1 if lado_direito else 0, xref="paper", y=0.5, yref="paper", showarrow=False,
            xanchor="right" if lado_direito else "left",
            text=(f"{rotulo_h0} →" if lado_direito else f"← {rotulo_h0}") + "<br>(fora da escala)",
        )
    fig = layout_padrao(fig, "", "", altura=220)
    fig.update_xaxes(range=faixa, showgrid=True)
    fig.update_yaxes(showticklabels=False)
    return legenda_referencia(fig)


def cartao_hipotese(titulo: str, r: dict, fmt, fmt_dif, simbolo_param: str, descricao_md: str,
                    condicoes: list[tuple[str, bool]], rotulo_h0: str) -> None:
    """fmt formata valores do parâmetro; fmt_dif formata diferenças e margens de erro."""
    with st.container(border=True):
        st.subheader(titulo)
        alt = r["alternativa"]
        v0 = fmt(r["valor0"])
        col1, col2 = st.columns([1, 1])
        with col1:
            markdown(descricao_md)
            markdown(f"**Hipóteses**\n\n"
                        f"- H₀: {simbolo_param} {SIMBOLO_H0[alt]} {v0}\n"
                        f"- H₁: {simbolo_param} {SIMBOLO_H1[alt]} {v0}")
            markdown("**Condições de aplicação**\n\n" + "\n".join(
                f"- {'✅' if ok else '⚠️'} {texto}" for texto, ok in condicoes))
        with col2:
            c = st.columns(2)
            c[0].metric("Estimativa amostral", fmt(r["estimativa"]))
            c[1].metric("Parâmetro da população", fmt(r["pop"]))
            c = st.columns(2)
            c[0].metric(f"Estatística de teste ({r['nome_estat']})", br(r["estatistica"], 3))
            c[1].metric("Valor-p", fmt_p(r["p_valor"]))
            st.dataframe(pd.DataFrame({
                "Item": ["Nível de significância (α)", "Intervalo de confiança 95%",
                         "Margem de erro (95%)", "Região de rejeição",
                         "Decisão"],
                "Valor": [br(ALFA, 2), f"[{fmt(r['ic_inf'])}; {fmt(r['ic_sup'])}]",
                          "± " + fmt_dif(r["margem"]), f"{r['nome_estat']} {r['regiao']}",
                          "Rejeita H₀" if r["rejeita"] else "Não rejeita H₀"],
            }), hide_index=True, width="stretch")

        st.plotly_chart(grafico_ic(r, fmt, rotulo_h0), width="stretch")

        # --- Conclusão do teste ---
        h1_txt = f"{simbolo_param} {SIMBOLO_H1[alt]} {v0}"
        if r["rejeita"]:
            conclusao = (f"Como o valor-p ({fmt_p(r['p_valor'])}) é menor que α = {br(ALFA, 2)}, "
                         f"**rejeita-se H₀**. Ao nível de 5%, há evidência estatística de que {h1_txt}.")
        else:
            conclusao = (f"Como o valor-p ({fmt_p(r['p_valor'])}) é maior que α = {br(ALFA, 2)}, "
                         f"**não se rejeita H₀**. A amostra não traz evidência de que {h1_txt}.")
            oposto = ((alt == "greater" and r["estimativa"] < r["valor0"]) or
                      (alt == "less" and r["estimativa"] > r["valor0"]))
            if oposto:
                lado = "abaixo" if alt == "greater" else "acima"
                conclusao += (f" A estimativa ({fmt(r['estimativa'])}) ficou {lado} de {v0}, ou seja, "
                              "do lado oposto ao da hipótese alternativa: os dados apontam na direção "
                              "contrária à de H₁.")
            conclusao += " Não rejeitar H₀ não prova que H₀ é verdadeira; apenas não há evidência contra ela."
        markdown("**Conclusão do teste**\n\n" + conclusao)

        # --- Comparação com a população ---
        dif = r["estimativa"] - r["pop"]
        dentro = r["ic_inf"] <= r["pop"] <= r["ic_sup"]
        h1_pop = h1_verdadeira_na_populacao(r)
        if r["rejeita"] == h1_pop:
            acerto = "a decisão do teste coincide com a realidade da população"
        elif r["rejeita"]:
            acerto = "o teste cometeu um **erro tipo I** (rejeitou uma H₀ verdadeira)"
        else:
            acerto = "o teste cometeu um **erro tipo II** (não rejeitou uma H₀ falsa)"
        markdown(
            "**Comparação com a população**\n\n"
            f"- A estimativa amostral ({fmt(r['estimativa'])}) difere do parâmetro populacional "
            f"({fmt(r['pop'])}) em {fmt_dif(dif)} ({br(dif / r['pop'] * 100, 2)}% em termos relativos).\n"
            f"- O parâmetro populacional **{'está dentro' if dentro else 'ficou fora'}** do "
            f"IC 95% [{fmt(r['ic_inf'])}; {fmt(r['ic_sup'])}]"
            + (": a diferença é compatível com o erro amostral (margem de ± "
               f"{fmt_dif(r['margem'])})." if dentro else
               ". Isso é esperado em cerca de 5% das amostras possíveis.") + "\n"
            f"- Na população, {simbolo_param} = {fmt(r['pop'])}, portanto "
            f"{'H₁' if h1_pop else 'H₀'} é a hipótese verdadeira: {acerto}."
        )


def painel_inferencia(arq_amostra: Path, arq_pop: Path) -> None:
    for arq in (arq_amostra, arq_pop):
        if not arq.exists():
            st.error(f"Arquivo não encontrado: {arq}. Rode ScriptParaBaixar.py antes.")
            return
    fonte_am = (arq_amostra.as_posix(), arq_amostra.stat().st_mtime)
    fonte_pop = (arq_pop.as_posix(), arq_pop.stat().st_mtime)

    am = dados_amostra(fonte_am)
    pop = indicadores(fonte_pop)
    n, N = len(am), int(pop["total"])

    st.header("Inferência estatística")
    legenda(f"Os procedimentos de inferência usam apenas a amostra (n = {br(n, 0)}). "
               f"Os parâmetros populacionais são calculados diretamente sobre as "
               f"{br(N, 0)} admissões da população, apenas para comparação. "
               f"Nível de significância α = {br(ALFA, 2)}, correspondente a 95% de confiança.")

    fmt_prop = lambda v: pct(v, 2)
    fmt_real = lambda v: "R$ " + br(v, 2)
    fracao = n / N
    cond_aas = ("Amostra aleatória simples sem reposição, sorteada da população (semente 2025)", True)
    cond_indep = (f"Independência: n é {pct(fracao, 3)} da população (< 5%)", fracao < 0.05)

    # --- a) Proporção -------------------------------------------------
    hp = HIP_PROPORCAO
    rp = inferencia_proporcao((am["sexo"] == 3).astype(int), hp["p0"], hp["alternativa"],
                              pop["prop_mulheres"])
    cartao_hipotese(
        "a) Proporção de mulheres entre os admitidos", rp, fmt_prop, dif_pp, "p",
        "**Variável:** sexo (qualitativa nominal)  \n"
        "**Evento de interesse:** o admitido ser mulher  \n"
        "**Parâmetro:** p = proporção de mulheres entre as admissões formais de 2025  \n"
        "**Justificativa:** as mulheres são cerca de metade da população brasileira em idade de "
        f"trabalhar. O teste verifica se elas representam "
        f"{SENTIDO[hp['alternativa']]} {pct(hp['p0'], 0)} das contratações com carteira assinada, "
        "ou seja, se há sub-representação feminina nas admissões formais."
        if hp["alternativa"] == "less" else
        f"trabalhar. O teste verifica se elas representam "
        f"{SENTIDO[hp['alternativa']]} {pct(hp['p0'], 0)} das contratações com carteira assinada.",
        [cond_aas, cond_indep,
         (f"n·p₀ = {br(n * hp['p0'], 1)} ≥ 10 e n·(1 − p₀) = {br(n * (1 - hp['p0']), 1)} ≥ 10 "
          "(aproximação normal válida)", min(n * hp["p0"], n * (1 - hp["p0"])) >= 10)],
        f"H₀: p₀ = {pct(hp['p0'], 0)}",
    )

    # --- b) Média -----------------------------------------------------
    hm = HIP_MEDIA
    rm = inferencia_media(am["salario"], hm["mu0"], hm["alternativa"], pop["sal_medio"])
    cartao_hipotese(
        "b) Salário médio de admissão", rm, fmt_real, dif_real, "μ",
        "**Variável:** salário de admissão (quantitativa contínua, R$/mês)  \n"
        "**Parâmetro:** μ = salário médio das admissões formais de 2025  \n"
        f"**Justificativa:** compara o salário de quem foi contratado em 2025 com o "
        f"{hm['fonte_mu0']} ({fmt_real(hm['mu0'])}), verificando se as novas contratações "
        f"pagam, em média, {SENTIDO[hm['alternativa']]} essa referência.",
        [cond_aas, cond_indep,
         (f"n = {br(n, 0)} ≥ 30 e σ desconhecido: teste t de Student", n >= 30),
         (f"A distribuição do salário é assimétrica (assimetria = {br(rm['assimetria'], 2)}), "
          "mas, pelo Teorema Central do Limite, com n grande a média amostral tem "
          "distribuição aproximadamente normal", n >= 30)],
        f"H₀: μ₀ = {fmt_real(hm['mu0'])}",
    )

    # --- Comparação geral ----------------------------------------------
    st.subheader("Comparação entre população e amostra")
    linhas = []
    for nome, r, fmt, fmt_dif in [("Proporção de mulheres", rp, fmt_prop, dif_pp),
                                  ("Salário médio", rm, fmt_real, dif_real)]:
        dif = r["estimativa"] - r["pop"]
        linhas.append({
            "Parâmetro": nome,
            "População": fmt(r["pop"]),
            "Amostra (estimativa)": fmt(r["estimativa"]),
            "Diferença": fmt_dif(dif),
            "Margem de erro": "± " + fmt_dif(r["margem"]),
            "Diferença relativa": br(dif / r["pop"] * 100, 2) + "%",
            "IC 95%": f"[{fmt(r['ic_inf'])}; {fmt(r['ic_sup'])}]",
            "Parâmetro no IC?": "Sim" if r["ic_inf"] <= r["pop"] <= r["ic_sup"] else "Não",
            "Hipótese testada": f"H₁: {SIMBOLO_H1[r['alternativa']]} {fmt(r['valor0'])}",
            "Estatística": f"{r['nome_estat']} = {br(r['estatistica'], 3)}",
            "Valor-p": fmt_p(r["p_valor"]),
            "Decisão": "Rejeita H₀" if r["rejeita"] else "Não rejeita H₀",
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

    markdown(
        "**Estimativas pontuais e intervalos de confiança.** A estimativa pontual é o valor "
        "mais provável do parâmetro segundo a amostra, mas quase nunca coincide exatamente com "
        "ele. O intervalo de confiança acrescenta a incerteza: o procedimento produz intervalos "
        "que contêm o parâmetro verdadeiro em 95% das amostras possíveis. Aqui, as estimativas "
        f"diferem dos parâmetros em {dif_pp(abs(rp['estimativa'] - rp['pop']))} (proporção) e "
        f"{dif_real(abs(rm['estimativa'] - rm['pop']))} (média), "
        + ("diferenças menores que as margens de erro"
           if abs(rp["estimativa"] - rp["pop"]) <= rp["margem"]
           and abs(rm["estimativa"] - rm["pop"]) <= rm["margem"]
           else "a serem comparadas com as margens de erro")
        + f" (± {dif_pp(rp['margem'])} e ± {dif_real(rm['margem'])}).\n\n"
        "**Testes unilaterais e IC bilateral.** Os testes são unilaterais a 5%, enquanto os "
        "intervalos são bilaterais a 95%. A decisão do teste equivale a verificar se o valor de H₀ "
        "fica além do limite unilateral de 95% (o mesmo que um IC bilateral de 90%); por isso, "
        "em casos limítrofes, o IC de 95% e o teste podem parecer discordar."
    )

    st.subheader("Limitações e variabilidade amostral")
    markdown(
        f"- **Variabilidade amostral:** os resultados vêm de uma única amostra. Outra semente "
        "geraria outras estimativas, outros intervalos e, em casos limítrofes, outra decisão. "
        "Em média, 5% dos intervalos de 95% não contêm o parâmetro.\n"
        f"- **Significância estatística × relevância prática:** com n = {br(n, 0)}, o erro-padrão "
        "é pequeno e diferenças pequenas já se tornam estatisticamente significativas. O valor-p "
        "não mede o tamanho do efeito; por isso as diferenças são mostradas também em valores "
        "absolutos e relativos.\n"
        "- **Comparabilidade da referência do salário:** o valor do IBGE abrange todos os "
        "ocupados (inclusive informais, autônomos e trabalhadores com muitos anos de casa), enquanto "
        "o CAGED registra apenas o salário contratual de admissões com carteira assinada. Salários "
        "de entrada tendem a ser menores, o que deve ser considerado na interpretação.\n"
        "- **Assimetria do salário:** a distribuição é muito assimétrica à direita. O teste t é "
        "válido pelo tamanho da amostra, mas a média é sensível a valores altos; a mediana "
        "(Painéis 1 e 2) complementa a análise.\n"
        "- **Limpeza dos dados:** a população analisada exclui registros inválidos e outliers "
        "(salário fora do intervalo P1–P99,9 e horas abaixo do P1). Os parâmetros dependem "
        "dessas regras.\n"
        "- **Cobertura do CAGED:** só inclui o emprego formal regido pela CLT; as conclusões não "
        "se estendem ao mercado de trabalho como um todo."
    )


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------
st.set_page_config(page_title="Painel CAGED", layout="wide")
st.title("Painel CAGED")

painel1, painel2, painel3 = st.tabs(["Painel 1", "Painel 2", "Painel 3"])

LIMPEZA = ("após limpeza e remoção de outliers "
           "(salário entre P1 e P99,9; horas a partir do P1).")

with painel1:
    painel_descritivo(
        ARQ_POP, "Análise descritiva da população",
        f"Todas as admissões do Novo CAGED em 2025, {LIMPEZA}",
    )

with painel2:
    painel_descritivo(
        ARQ_AMOSTRA, "Análise descritiva da amostra",
        "Amostra aleatória simples sem reposição sorteada da população pelo "
        "ScriptParaBaixar.py (semente 2025, 95% de confiança, margem de erro de 1 p.p.). "
        f"A população foi obtida das admissões do Novo CAGED em 2025, {LIMPEZA}",
        referencia=ARQ_POP,
    )

with painel3:
    painel_inferencia(ARQ_AMOSTRA, ARQ_POP)
