# Relatório Técnico — Assistente Médico Virtual (Tech Challenge Fase 3)

**Pós-Graduação IA Para Desenvolvedores · FIAP**

---

## 1. Contexto e Objetivo

O hospital solicitou um assistente virtual médico treinado com dados
internos (protocolos, FAQs de médicos, modelos de laudos/receitas),
capaz de auxiliar condutas clínicas, responder dúvidas de médicos e
sugerir procedimentos com base em protocolos internos — sempre sob
validação humana — além de fluxos de decisão automatizados e seguros
coordenados com LangChain/LangGraph.

Este relatório documenta o processo de fine-tuning, a arquitetura do
assistente, a avaliação do modelo e a análise dos resultados.

## 2. Processo de Fine-tuning

### 2.1 Dados de Origem

Três fontes internas sintéticas foram criadas em `data/raw/`, complementadas
por uma amostra real do dataset público **PubMedQA** (sugerido no enunciado
do desafio):

| Fonte | Conteúdo | Exemplos gerados |
|-------|----------|:-----------------:|
| `protocolos/*.md` | 3 protocolos clínicos completos (dor torácica, sepse, hiperglicemia) — sintéticos | 17 |
| `faqs/faqs_medicos.jsonl` | 10 perguntas frequentes de médicos com resposta e fonte — sintéticas | 10 |
| `laudos/modelos_laudos.md` | 4 modelos estruturados (ECG, receita, evolução, solicitação de exame) — sintéticos | 4 |
| `pubmedqa/pubmedqa_sample.jsonl` | Amostra real do [PubMedQA](https://pubmedqa.github.io/) (Jin et al., 2019) — perguntas de pesquisa biomédica com resposta longa e decisão (yes/no/maybe), baixada via Hugging Face Hub (`qiaojin/PubMedQA`, config `pqa_labeled`) e **traduzida para PT-BR** (`Helsinki-NLP/opus-mt-tc-big-en-pt`, MarianMT local) por `src/data_processing/download_pubmedqa.py` | 40 |

O original em inglês é preservado em `pubmedqa_sample_en.jsonl` para
auditoria da tradução — todo o restante do dataset e das respostas do
assistente permanece em português, consistente com o idioma do projeto.

O PubMedQA amplia o escopo do assistente para além dos protocolos internos:
perguntas clínicas gerais baseadas em publicações médicas reais também são
respondidas, sempre com a fonte `PubMedQA (pubmedqa.github.io)` citada
separadamente das fontes internas do hospital.

### 2.2 Preprocessing, Anonimização e Curadoria

`src/data_processing/anonymize.py` processa `data/raw/prontuarios_brutos.csv`
(que contém PII sintética — nome, CPF, telefone, e-mail, endereço,
convênio) e produz `data/processed/prontuarios_anonimizados.csv`, com:

1. **Remoção estrutural** das 6 colunas identificadoras diretas.
2. **Varredura por regex** (segunda camada) em campos de texto livre
   remanescentes, mascarando CPF, telefone e e-mail residuais.
3. Manutenção apenas do `paciente_id` (pseudônimo) como chave de junção.

`src/data_processing/dataset_builder.py` executa a curadoria: fragmenta
(chunking) os protocolos por seção, converte as FAQs e os modelos de
laudo em pares instrução/resposta, e persiste tudo em
`data/processed/finetune_dataset.jsonl` — **71 exemplos** no formato
instruction-tuning (estilo Alpaca).

### 2.3 Pipeline de Treinamento

`src/finetuning/train.py` realiza fine-tuning supervisionado com
**LoRA (Low-Rank Adaptation)** via `peft`. O modelo base **padrão** é
`distilgpt2` (82M parâmetros) — treina em ~90 segundos em qualquer CPU,
escolha deliberada para que `py -m src.main finetune` seja reproduzível
por qualquer avaliador sem espera longa.

O pipeline também **suporta** trocar o modelo base para uma arquitetura
LLaMA real via `--base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0`
(`LlamaForCausalLM`, 1.1B parâmetros, aberto no Hugging Face — sem exigir
aceite de licença/token, ao contrário do Llama 3 oficial da Meta, que é
*gated*). Os módulos do LoRA são **auto-detectados por família de
arquitetura** (`default_lora_target_modules()`): `c_attn` para GPT-2,
`q_proj/k_proj/v_proj/o_proj` para LLaMA/Mistral. Tentamos essa execução
nesta máquina, apenas em CPU (sem GPU), e **após mais de 6 horas de
processamento sem completar sequer 1 dos 10 passos configurados, o
computador desligou sozinho** por sobrecarga — por isso o TinyLlama não é
o padrão do projeto nem consta aqui como resultado concluído; a equipe
seguiu com o `distilgpt2` como modelo padrão. A capacidade fica
registrada como suportada e verificável por quem tiver GPU disponível,
inclusive com scripts de instalação do PyTorch para NVIDIA/AMD já
prontos em `scripts/`.

Hiperparâmetros (ver `src/finetuning/config.py`): `r=8`, `alpha=16`,
`dropout=0.05`, 3 épocas, batch size 2, learning rate `2e-4`.

O adaptador treinado é salvo em `artifacts/finetuned_model/`, junto com
`training_manifest.json` (metadados reprodutíveis: modelo base, número
de exemplos, hiperparâmetros).

> **Nota de execução:** as dependências de treino (`torch`, `transformers`,
> `peft`, `datasets`) são opcionais e pesadas — ver
> `requirements-finetuning.txt`. O restante do sistema (RAG, LangChain,
> LangGraph, segurança) roda de forma independente, usando
> `DeterministicMedicalProvider` como fallback determinístico e
> auditável, o que garante testes 100% reprodutíveis sem GPU.

### 2.4 Avaliação do Modelo

`src/finetuning/evaluate.py` separa um conjunto de validação (hold-out
das últimas 3 FAQs, não usadas no treino) e mede:

- **Overlap lexical** entre resposta gerada e resposta de referência —
  métrica simples e auditável, sem dependências pesadas.
- **Perplexidade** no conjunto de validação, quando o modelo fine-tuned
  está disponível (via `transformers`).

Resultado do fallback determinístico no hold-out (baseline de
comparação, artefato em `artifacts/eval_report.json`):

| Métrica | Resultado |
|---------|:---------:|
| Overlap lexical médio | **1.0** (3/3 perguntas de hold-out corretamente recuperadas da base curada) |
| Perguntas avaliadas | 3 |

Esse resultado representa o piso de qualidade do sistema — mesmo sem o
modelo fine-tuned carregado, o assistente responde corretamente com base
na curadoria feita. O modelo fine-tuned é avaliado com a mesma interface
(`answer_question`), permitindo comparação direta assim que treinado.

## 3. Descrição do Assistente Médico

`src/langchain_app/medical_assistant.py` implementa `MedicalAssistant`,
que integra em um único pipeline:

1. **RAG** sobre os protocolos internos (`src/rag/`) — TF-IDF local por
   padrão, ou FAISS via LangChain quando as dependências opcionais estão
   instaladas.
2. **Consulta à base estruturada de prontuários** (SQLite, já
   anonimizada) via `src/database/patient_repository.py`.
3. **Geração de resposta** pela LLM (fine-tuned local, OpenAI ou
   determinística — `src/llm/provider.py`).
4. **Guardrails de segurança** (`src/safety/guardrails.py`).
5. **Explainability** — toda resposta cita as fontes usadas e o contexto
   do paciente considerado (`src/explainability/citations.py`).
6. **Logging de auditoria** (`src/safety/audit_logger.py`).

O fluxo de decisão automatizado (`src/langgraph_flow/clinical_flow.py`)
coordena, via LangGraph, as etapas: `intake` → `verificar_exames_pendentes`
→ `consultar_protocolo` → `sugerir_conduta` → `revisao_seguranca` →
`emitir_alertas`. Ver diagrama completo em `docs/architecture.md`.

## 4. Segurança e Limites de Atuação

- O assistente **nunca prescreve diretamente**: `MedicalGuardrails.review_output`
  detecta e reescreve linguagem de ordem médica direta (`administre`,
  `prescrevo`, `injete`...), e todo texto final recebe o aviso obrigatório
  `[AVISO] ... requer validação de um médico responsável`.
- Perguntas fora do escopo seguro (`check_input`) são bloqueadas antes de
  qualquer geração de resposta.
- Todo evento (pergunta, paciente, fontes, flags) é persistido em
  `logs/audit_log.jsonl`, com `event_id` único, timestamp UTC e
  rastreabilidade completa — atendendo ao requisito de logging detalhado
  e auditoria.

## 5. Organização do Código

Projeto modularizado em `src/`, com pacotes de responsabilidade única:
`data_processing`, `finetuning`, `llm`, `rag`, `database`,
`langchain_app`, `langgraph_flow`, `safety`, `explainability`. CLI única
(`src/main.py`) expõe todos os comandos. 26 testes automatizados
(`tests/`, `pytest`) cobrem anonimização, curadoria, RAG, guardrails,
prontuário, assistente e fluxo clínico.

## 6. Limitações e Trabalhos Futuros

- O modelo base usado por padrão (`distilgpt2`) é pequeno, escolhido por
  reprodutibilidade em CPU sem GPU; o pipeline suporta uma arquitetura
  LLaMA real (TinyLlama-1.1B) via `--base-model`, mas uma tentativa de
  execução nesta máquina, só em CPU, passou de 6 horas de processamento
  sem completar sequer 1 dos 10 passos configurados e terminou com o
  computador desligando sozinho por sobrecarga — inviável como padrão
  sem aceleração de hardware. Para uso real em produção, recomenda-se um
  modelo de 7B+
  parâmetros (Llama 3, Mistral)
  com infraestrutura de GPU dedicada.
- Os protocolos e prontuários são sintéticos, criados para fins
  acadêmicos — em produção, exigiriam curadoria por equipe médica e
  aprovação de comitê de ética/privacidade (LGPD).
- O RAG padrão (TF-IDF) é intencionalmente leve; a versão com embeddings
  semânticos via FAISS/LangChain (`requirements-langchain.txt`) melhora a
  qualidade de recuperação em bases maiores.
