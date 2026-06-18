# Caracterização de carga e geração de log sintético a partir do AOL Query Log

Este repositório reúne o código do Trabalho de Conclusão de Curso (TCC) em Sistemas de Informação na Universidade Federal de Santa Catarina (UFSC). O trabalho caracteriza o padrão de carga de um servidor de busca usando o *AOL Query Log* e, a partir dessa caracterização, gera um *log* sintético no mesmo formato dos dados reais.

O objetivo não é gerar carga diretamente. O artefato final é um *log* sintético que pode ser usado depois como entrada para uma ferramenta geradora de carga.

- **Conjunto de dados:** *AOL Query Log*, segmento `user-ct-test-collection-02`, período de março a maio de 2006

## O que o projeto faz

O trabalho está dividido em quatro partes que se encadeiam:

1. **Caracterização estatística** do padrão de requisições por hora, incluindo distribuição de popularidade de consultas, recorrência por usuário, tempo entre chegadas (*interarrival time*) e autocorrelação.
2. **Segmentação de perfis de carga** com *K-Means* (k = 5), agrupando as janelas horárias em perfis com volumes e comportamentos parecidos.
3. **Detecção de anomalias** com *DBSCAN*, identificando janelas que fogem dos perfis típicos.
4. **Previsão de volume** com um modelo *SARIMA*, usado como fonte paramétrica do número de requisições por hora.

No fim, o gerador (`gerador_log.py`) combina a previsão do *SARIMA* com os perfis do *K-Means* para produzir registros sintéticos no formato AOL (`AnonID`, `Query`, `QueryTime`, `ItemRank`, `ClickURL`).

## Conjunto de dados

O *AOL Query Log* não está incluído no repositório por questões de tamanho e de uso. Ele precisa ser baixado à parte e colocado na pasta de dados brutos. Cada linha original traz o identificador anônimo do usuário, o texto da consulta, o instante da requisição e, quando houve clique, a posição e a URL do resultado clicado.

O *dataset* está disponível no [Kaggle](https://www.kaggle.com/datasets/dineshydv/aol-user-session-collection-500k).


## Pipeline

Os *scripts* foram pensados para rodar nesta ordem. Cada etapa lê o arquivo gerado pela anterior.

### 1. Preparação dos dados

| Script | O que faz |
|--------|-----------|
| `converter.py` | Converte o arquivo `.txt` original para CSV, validando o formato de cada linha. |
| `clean_csv.py` | Remove linhas inválidas e duplicatas (registros idênticos e mesmo usuário no mesmo instante). |
| `remove_incomplete.py` | Descarta dias com cobertura menor que 18 horas, para não distorcer a análise temporal. |

### 2. Caracterização

| Script | O que faz |
|--------|-----------|
| `characterize_aol_general_filtered.py` | Estatísticas gerais (requisições por usuário, por dia, taxa de clique). |
| `characterize_aol_query_popularity.py` | Distribuição de popularidade das consultas. |
| `characterize_aol_query_recurrence.py` | Recorrência de consultas por usuário. |
| `characterize_aol_interarrival.py` | Tempo entre chegadas, global e por usuário. |
| `characterize_aol_time_buckets.py` | Agregação por faixas de tempo. |
| `characterize_aol_acf.py` | Função de autocorrelação em diferentes granularidades. |
| `decomposicao.py` | Decomposição da série temporal em tendência, sazonalidade e resíduo. |

### 3. Perfis de carga e anomalias

| Script | O que faz |
|--------|-----------|
| `features.py` | Monta as *features* por janela horária usadas no agrupamento. |
| `kmeans.py` | Gera os cinco perfis de carga e as métricas de avaliação (cotovelo e *silhouette*). |
| `dbscan.py` | Identifica janelas anômalas (ruído) a partir das mesmas *features*. |
| `analise_perfil_1.py` | Análise detalhada de um perfil específico. |

### 4. Previsão com SARIMA

Os *scripts* numerados de `01` a `06` formam a etapa de modelagem da série temporal:

| Script | O que faz |
|--------|-----------|
| `01_preparar_serie.py` | Monta a série horária e separa treino, validação e teste. |
| `02_diagnostico_estacionariedade.py` | Testes ADF e KPSS sobre a série e suas diferenciações. |
| `03_acf_pacf.py` | Gráficos de ACF e PACF para apoiar a escolha das ordens. |
| `04_ajustar_modelos_validacao.py` | Ajuste dos modelos candidatos e comparação na validação. |
| `05_treinar_melhor_modelo.py` | Treino do modelo escolhido e avaliação no teste, com *baselines*. |
| `06_previsoes.py` | Previsão para o horizonte futuro. |

### 5. Geração do log sintético

| Script | O que faz |
|--------|-----------|
| `gerador_log.py` | Gera o *log* sintético combinando a previsão do *SARIMA* com os perfis do *K-Means*. |

## Como executar

O projeto usa Python 3.10 ou superior e as bibliotecas abaixo:

```
pandas
numpy
scikit-learn
statsmodels
matplotlib
joblib
```

Instalação:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib joblib
```

Com o arquivo bruto do AOL já na pasta de dados, a sequência é:

```bash
python converter.py
python clean_csv.py
python remove_incomplete.py
python features.py
python kmeans.py
python dbscan.py
python 01_preparar_serie.py
python 02_diagnostico_estacionariedade.py
python 03_acf_pacf.py
python 04_ajustar_modelos_validacao.py
python 05_treinar_melhor_modelo.py
python 06_previsoes.py
python gerador_log.py
```

O gerador aceita dois parâmetros opcionais:

```bash
python gerador_log.py --forecast caminho/para/previsao.csv --saida caminho/para/log_sintetico.csv
```

O arquivo de previsão precisa ter as colunas `window_start` e `forecast_sarima` (ou equivalente). Por padrão, o gerador usa a previsão do *SARIMA* para o conjunto de teste. Para gerar o *log* de uma única faixa de hora, basta criar um CSV com essa mesma estrutura.

Além do *log* em si, o gerador salva um arquivo de validação que reconstrói as *features* por hora a partir dos registros sintéticos e as compara com o volume previsto pelo *SARIMA*.

## Estrutura esperada das pastas

Os *scripts* usam caminhos relativos e assumem uma organização parecida com esta:

```
AOL Query Log/
├── raw data/                 # arquivo .txt original do AOL
├── processed/                # CSVs limpos
├── outputs txt/              # resumos em texto das etapas
├── analise_treino/           # caracterização e perfis (subconjunto de treino)
│   ├── features/
│   └── perfis_carga_kmeans/
├── analise_completa/         # série temporal completa
│   └── sarima/
├── log_sintetico/            # saída do gerador
└── scripts/                  # os scripts deste repositório
```

Os caminhos podem ser ajustados no início de cada *script*, nas variáveis de configuração.

## Saída

O *log* sintético segue o formato do AOL original:

| Coluna | Descrição |
|--------|-----------|
| `AnonID` | Identificador anônimo do usuário. |
| `Query` | Texto da consulta (sintético). |
| `QueryTime` | Instante da requisição. |
| `ItemRank` | Posição do resultado clicado, quando houve clique. |
| `ClickURL` | URL clicada, quando houve clique. |

