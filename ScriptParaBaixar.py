"""
Novo CAGED (admissões) - extração dos .7z, remoção de outliers,
estatística descritiva, amostragem (AAS) e intervalos de confiança.

Uso:
    pip install py7zr duckdb pandas numpy scipy pyarrow
    python analise_caged.py            # usa a pasta ./2025
    python analise_caged.py D:/dados/2025
"""

import sys
import math
import unicodedata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import py7zr
from scipy import stats

# ----------------------------------------------------------------------
# 1. CONFIGURAÇÕES (ajuste aqui)
# ----------------------------------------------------------------------
PASTA_SCRIPT = Path(__file__).resolve().parent  # caminhos relativos à pasta do script
PASTA_ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_SCRIPT / "2025"
PASTA_EXTRAIDOS = PASTA_ENTRADA / "extraidos"   # onde os .txt serão gravados
PASTA_SAIDA = PASTA_SCRIPT / "saida"            # parquet/csv para os painéis
TIPO_ARQUIVO = "CAGEDMOV"                       # MOV = admissões e desligamentos

SEMENTE = 2025                  # semente da AAS (registre no relatório)
NIVEL_CONFIANCA = 0.95          # usado no tamanho amostral e nos ICs
MARGEM_ERRO = 0.01              # margem de erro do dimensionamento (1 p.p.)
P_DIMENSIONAMENTO = 0.5         # p = 0,5 => caso mais conservador

# Regras de limpeza
IDADE_MIN, IDADE_MAX = 14, 75
PERCENTIL_CORTE_SALARIO = 0.999  # remove o 0,1% de salários mais altos
PERCENTIL_CORTE_INFERIOR = 0.01  # remove o 1% de salários e de horas mais baixos

PASTA_SAIDA.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# 2. EXTRAÇÃO DOS ARQUIVOS .7Z
# ----------------------------------------------------------------------
def extrair_7z(pasta_entrada: Path, pasta_destino: Path, prefixo: str) -> list[Path]:
    """Procura os .7z (inclusive em subpastas) e extrai só os do tipo pedido."""
    pasta_destino.mkdir(parents=True, exist_ok=True)
    arquivos_7z = sorted(
        p for p in pasta_entrada.rglob("*.7z") if p.name.upper().startswith(prefixo)
    )
    if not arquivos_7z:
        raise FileNotFoundError(f"Nenhum {prefixo}*.7z encontrado em {pasta_entrada}")

    for arq in arquivos_7z:
        destino = pasta_destino / arq.stem
        if destino.exists() and any(destino.iterdir()):
            print(f"[extração] já extraído, pulando: {arq.name}")
            continue
        destino.mkdir(exist_ok=True)
        print(f"[extração] {arq.name}")
        with py7zr.SevenZipFile(arq, mode="r") as z:
            z.extractall(path=destino)

    return sorted(p for p in pasta_destino.rglob("*.txt") if p.name.upper().startswith(prefixo))


# ----------------------------------------------------------------------
# 3. LEITURA E LIMPEZA / REMOÇÃO DE OUTLIERS (DuckDB, para não estourar a memória)
# ----------------------------------------------------------------------
def sem_acento(texto: str) -> str:
    """'competênciamov' -> 'competenciamov' (nomes de coluna sem acento)."""
    t = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in t if not unicodedata.combining(c)).strip().lower()


def ler_cabecalho(arquivo: Path) -> tuple[list[str], str]:
    """Lê a 1ª linha; descobre se o arquivo é UTF-8 ou Latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            with open(arquivo, encoding=enc) as f:
                linha = f.readline()
            return [sem_acento(c) for c in linha.strip().split(";")], enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Não foi possível decodificar o cabeçalho")


def construir_populacao(txts: list[Path], saida_parquet: Path) -> None:
    colunas, enc = ler_cabecalho(txts[0])
    lista_arquivos = ", ".join(f"'{p.as_posix()}'" for p in txts)
    nomes = ", ".join(f"'{c}'" for c in colunas)

    con = duckdb.connect()

    # Tudo é lido como texto e convertido depois, porque os números usam vírgula decimal.
    con.execute(f"""
        CREATE VIEW bruto AS
        SELECT * FROM read_csv([{lista_arquivos}],
            delim=';', header=true, names=[{nomes}],
            all_varchar=true, encoding='{enc}', union_by_name=true)
    """)

    # Conversão de tipos + filtros de qualidade. O salário já vem convertido
    # para base mensal no CAGED (valorsalariofixo guarda o valor na unidade original).
    con.execute("""
        CREATE TABLE tipado AS
        SELECT
            TRY_CAST(competenciamov AS INTEGER)               AS competencia,
            TRY_CAST(regiao AS INTEGER)                       AS regiao,
            TRY_CAST(uf AS INTEGER)                           AS uf,
            municipio,
            secao,
            TRY_CAST(saldomovimentacao AS INTEGER)            AS saldo,
            TRY_CAST(graudeinstrucao AS INTEGER)              AS grau_instrucao,
            TRY_CAST(idade AS INTEGER)                        AS idade,
            TRY_CAST(REPLACE(horascontratuais, ',', '.') AS DOUBLE) AS horas,
            TRY_CAST(racacor AS INTEGER)                      AS raca_cor,
            TRY_CAST(sexo AS INTEGER)                         AS sexo,
            TRY_CAST(tipomovimentacao AS INTEGER)             AS tipo_mov,
            TRY_CAST(indtrabparcial AS INTEGER)               AS trab_parcial,
            TRY_CAST(REPLACE(salario, ',', '.') AS DOUBLE)    AS salario
        FROM bruto
    """)

    total = con.execute("SELECT COUNT(*) FROM tipado").fetchone()[0]

    # População de interesse: ADMISSÕES (saldo = +1) com dados válidos.
    base_valida = """
        FROM tipado
        WHERE saldo = 1
          AND salario > 0
          AND horas > 0
          AND municipio <> '999999'
          AND idade BETWEEN {imin} AND {imax}
          AND sexo IN (1, 3)
    """.format(imin=IDADE_MIN, imax=IDADE_MAX)

    # Cortes de outliers: limite superior do salário e limites inferiores
    # de salário e horas, todos calculados sobre a base válida.
    corte_sal_sup, corte_sal_inf, corte_horas_inf = con.execute(f"""
        SELECT quantile_cont(salario, {PERCENTIL_CORTE_SALARIO}),
               quantile_cont(salario, {PERCENTIL_CORTE_INFERIOR}),
               quantile_cont(horas,   {PERCENTIL_CORTE_INFERIOR})
        {base_valida}
    """).fetchone()

    con.execute(f"""
        COPY (
            SELECT * {base_valida}
              AND salario BETWEEN {corte_sal_inf} AND {corte_sal_sup}
              AND horas >= {corte_horas_inf}
        )
        TO '{saida_parquet.as_posix()}' (FORMAT PARQUET)
    """)
    final = con.execute(f"SELECT COUNT(*) FROM '{saida_parquet.as_posix()}'").fetchone()[0]

    print(f"[limpeza] linhas lidas: {total:,} | população final: {final:,}")
    print(f"[limpeza] salário mantido entre R$ {corte_sal_inf:,.2f} "
          f"(p{PERCENTIL_CORTE_INFERIOR*100:.1f}) e R$ {corte_sal_sup:,.2f} "
          f"(p{PERCENTIL_CORTE_SALARIO*100:.1f})")
    print(f"[limpeza] horas mantidas a partir de {corte_horas_inf:,.2f} "
          f"(p{PERCENTIL_CORTE_INFERIOR*100:.1f})")


# ----------------------------------------------------------------------
# 4. VARIÁVEIS DERIVADAS E DESCRIÇÃO
# ----------------------------------------------------------------------
# CONFIRA no dicionário de layout do CAGED se estes códigos estão corretos.
REGIOES = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}
SEXOS = {1: "Homem", 3: "Mulher"}
GRAU_SUPERIOR = {9, 10, 11, 80}   # superior completo, mestrado, doutorado, pós


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mulher"] = (df["sexo"] == 3).astype(int)
    df["superior"] = df["grau_instrucao"].isin(GRAU_SUPERIOR).astype(int)
    df["sexo_desc"] = df["sexo"].map(SEXOS)
    df["regiao_desc"] = df["regiao"].map(REGIOES).fillna("Não identificada")
    return df


def descritiva(df: pd.DataFrame, nome: str) -> None:
    quant = df[["salario", "idade", "horas"]].agg(
        ["count", "mean", "median", "std", "min", "max"]).T
    quant["cv_%"] = quant["std"] / quant["mean"] * 100
    quant.to_csv(PASTA_SAIDA / f"descritiva_quant_{nome}.csv")

    for col in ["sexo_desc", "regiao_desc", "uf", "secao", "grau_instrucao"]:
        freq = df[col].value_counts(dropna=False).rename("n").to_frame()
        freq["proporcao"] = freq["n"] / len(df)
        freq.to_csv(PASTA_SAIDA / f"freq_{col}_{nome}.csv")

    print(f"\n--- Descritiva ({nome}) ---")
    print(quant.round(2))


# ----------------------------------------------------------------------
# 5. DIMENSIONAMENTO E AAS
# ----------------------------------------------------------------------
def tamanho_amostra(N: int, conf: float, e: float, p: float) -> int:
    """n0 = z² p(1-p) / e²  ;  n = n0 / (1 + (n0 - 1)/N)  (correção para população finita)."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n0 = z**2 * p * (1 - p) / e**2
    return math.ceil(n0 / (1 + (n0 - 1) / N))


def selecionar_aas(df: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    """AAS sem reposição: sorteia n posições de 0..N-1, todas com igual probabilidade."""
    rng = np.random.default_rng(semente)
    posicoes = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(posicoes)].reset_index(drop=True)


# ----------------------------------------------------------------------
# 6. INTERVALOS DE CONFIANÇA
# ----------------------------------------------------------------------
def ic_proporcao(amostra, populacao, col, rotulo):
    n = len(amostra)
    p_hat = amostra[col].mean()
    p_pop = populacao[col].mean()
    z_crit = stats.norm.ppf(1 - (1 - NIVEL_CONFIANCA) / 2)

    # IC de Wald (p-hat usado no erro-padrão)
    ep = math.sqrt(p_hat * (1 - p_hat) / n)
    ic = (p_hat - z_crit * ep, p_hat + z_crit * ep)
    return {
        "parametro": rotulo, "tipo": "proporcao", "confianca": NIVEL_CONFIANCA,
        "valor_populacional": p_pop, "estimativa_amostral": p_hat,
        "erro_padrao": ep, "ic_inf": ic[0], "ic_sup": ic[1],
        "pop_dentro_do_IC": ic[0] <= p_pop <= ic[1], "n": n,
    }


def ic_media(amostra, populacao, col, rotulo):
    x = amostra[col]
    n = len(x)
    media, s = x.mean(), x.std(ddof=1)
    mu_pop = populacao[col].mean()

    # IC t de Student
    t_crit = stats.t.ppf(1 - (1 - NIVEL_CONFIANCA) / 2, df=n - 1)
    ep = s / math.sqrt(n)
    ic = (media - t_crit * ep, media + t_crit * ep)
    return {
        "parametro": rotulo, "tipo": "media", "confianca": NIVEL_CONFIANCA,
        "valor_populacional": mu_pop, "estimativa_amostral": media,
        "erro_padrao": ep, "ic_inf": ic[0], "ic_sup": ic[1],
        "pop_dentro_do_IC": ic[0] <= mu_pop <= ic[1], "n": n,
    }


# ----------------------------------------------------------------------
# 7. EXECUÇÃO
# ----------------------------------------------------------------------
def main():
    txts = extrair_7z(PASTA_ENTRADA, PASTA_EXTRAIDOS, TIPO_ARQUIVO)
    print(f"[extração] {len(txts)} arquivo(s) .txt prontos")

    parquet_pop = PASTA_SAIDA / "populacao.parquet"
    construir_populacao(txts, parquet_pop)

    pop = preparar(pd.read_parquet(parquet_pop))
    N = len(pop)
    pop.to_csv(PASTA_SAIDA / "populacao.csv", index=False)
    descritiva(pop, "populacao")

    n = tamanho_amostra(N, NIVEL_CONFIANCA, MARGEM_ERRO, P_DIMENSIONAMENTO)
    print(f"\n[amostra] N = {N:,} | confiança = {NIVEL_CONFIANCA:.0%} | "
          f"erro = {MARGEM_ERRO:.1%} | p = {P_DIMENSIONAMENTO} -> n = {n:,}")

    amostra = selecionar_aas(pop, n, SEMENTE)
    amostra.to_parquet(PASTA_SAIDA / "amostra.parquet")
    amostra.to_csv(PASTA_SAIDA / "amostra.csv", index=False)
    descritiva(amostra, "amostra")

    resultados = [
        ic_proporcao(amostra, pop, "mulher", "Proporção de mulheres admitidas"),
        ic_proporcao(amostra, pop, "superior", "Proporção com ensino superior"),
        ic_media(amostra, pop, "salario", "Salário médio de admissão"),
        ic_media(amostra, pop, "idade", "Idade média dos admitidos"),
    ]
    tabela = pd.DataFrame(resultados)
    tabela.to_csv(PASTA_SAIDA / "intervalos_confianca.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", None)
    print("\n--- Intervalos de confiança ---")
    print(tabela.round(4).T)

    # Tabela final: 5 primeiros registros com todas as variáveis (colunas)
    for nome, df in [("População", pop), ("Amostra", amostra)]:
        print(f"\n--- {nome}: 5 primeiros registros ({df.shape[1]} variáveis) ---")
        print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
