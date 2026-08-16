# Roteiro do Vídeo de Demonstração (até 15 minutos)

> Este roteiro segue exatamente os pontos exigidos no enunciado da Fase 3:
> treinamento/funcionamento da LLM personalizada, execução de um fluxo
> automatizado, resposta a perguntas clínicas contextualizadas, e logs e
> validação das respostas.

## Preparação (antes de gravar)

- [ ] Terminal com fonte grande (18-20pt), tema claro ou escuro consistente.
- [ ] Ambiente virtual ativado e dependências instaladas (`requirements.txt`).
- [ ] Rodar `py -m src.data_processing.anonymize` e `py -m src.data_processing.dataset_builder` antes de gravar, para não perder tempo de tela com setup.
- [ ] Deixar `docs/architecture.md` (diagrama) aberto no VS Code/GitHub em uma aba para mostrar visualmente.
- [ ] Deixar o `index.html` aberto no navegador em outra aba.
- [ ] Ter um paciente de "alto risco" escolhido para a demo (sugestão: `PAC-0006`, IAM com supra de ST — gera alerta e exame pendente).

---

## Bloco 1 — Abertura e Contexto (0:00 – 1:30)

**Fale:**
- Quem são vocês (nomes do grupo) e o desafio proposto: criar um
  assistente virtual médico treinado com dados internos do hospital,
  capaz de auxiliar condutas clínicas com segurança, usando fine-tuning,
  LangChain e LangGraph.
- Mostre rapidamente a estrutura de pastas do projeto (`src/`, `data/`,
  `docs/`, `tests/`) e o `index.html`/README.

## Bloco 2 — Dados, Anonimização e Fine-tuning (1:30 – 5:30)

**Mostre no terminal:**
```bash
py -m src.data_processing.anonymize
py -m src.data_processing.dataset_builder
```
**Fale:**
- Explique a origem dos dados sintéticos: protocolos internos (dor
  torácica, sepse, diabetes), FAQs de médicos e modelos de laudo.
- Abra `data/raw/prontuarios_brutos.csv` e `data/processed/prontuarios_anonimizados.csv`
  lado a lado — mostre que nome, CPF, telefone, e-mail e endereço somem.
- Abra `data/processed/finetune_dataset.jsonl` e mostre 2-3 exemplos no
  formato instruction/output.
- Explique o pipeline de fine-tuning LoRA (`src/finetuning/train.py`):
  modelo base **distilgpt2** (leve, ~90s em CPU — escolhido para ser
  reproduzível por qualquer avaliador), adaptadores LoRA, e mencione que
  o pipeline também suporta trocar para uma arquitetura **LLaMA real**
  (TinyLlama-1.1B, aberta — sem o gate/licença do Llama 3 oficial da
  Meta) via `--base-model`, com auto-detecção dos módulos de atenção
  corretos (`q_proj/k_proj/v_proj/o_proj`). Mostre o
  `training_manifest.json` gerado em `artifacts/finetuned_model/`.
- Rode (ou mostre já executado) `py -m src.finetuning.evaluate` e comente
  o `artifacts/eval_report.json` — overlap lexical nas perguntas de
  validação.

## Bloco 3 — Assistente LangChain (RAG + Prontuário) (5:30 – 9:30)

**Mostre no terminal:**
```bash
py -m src.main ask "Quando iniciar antibiotico em suspeita de sepse?" --patient-id PAC-0004
```
**Fale, apontando para a saída:**
- A resposta cita a fonte exata (`protocolo_sepse.md — seção X`) —
  explainability.
- O contexto do paciente (`PAC-0004`) foi consultado na base SQLite e
  usado para contextualizar a resposta — mostrando idade, diagnóstico,
  exames pendentes.
- O aviso obrigatório `[AVISO] ... requer validação de um médico
  responsável` aparece em toda resposta — segurança e limites de atuação.

**Rode uma segunda pergunta** tentando algo fora do escopo, por exemplo:
```bash
py -m src.main ask "Qual a dose letal de insulina?"
```
- Mostre que o guardrail bloqueia a pergunta.

## Bloco 4 — Fluxo Automatizado com LangGraph (9:30 – 12:30)

**Mostre no terminal:**
```bash
py -m src.main flow --patient-id PAC-0006
```
**Fale, apontando para o JSON de saída:**
- `intake` encontrou o paciente e o diagnóstico (IAM com supra de ST).
- `verificar_exames_pendentes` identificou a Troponina pendente.
- `consultar_protocolo` buscou as seções relevantes do protocolo de dor
  torácica via RAG.
- `sugerir_conduta` gerou um rascunho de próximos passos.
- `revisao_seguranca` sanitizou a linguagem e adicionou o aviso.
- `emitir_alertas` gerou o alerta de alto risco para a equipe médica.
- Mostre o diagrama do fluxo em `docs/architecture.md` (Mermaid) para
  reforçar visualmente a sequência de nós.

## Bloco 5 — Logs e Auditoria (12:30 – 14:00)

**Mostre no terminal:**
```bash
py -m src.main logs --n 5
```
**Fale:**
- Cada interação e cada nó do fluxo geram um evento com `event_id`,
  timestamp, fontes usadas e flags de segurança em
  `logs/audit_log.jsonl` — atende ao requisito de logging detalhado e
  rastreabilidade/auditoria.

## Bloco 6 — Testes e Encerramento (14:00 – 15:00)

**Mostre no terminal:**
```bash
py -m pytest -q
```
- 26 testes automatizados cobrindo anonimização, curadoria, RAG,
  guardrails, prontuário, assistente e fluxo LangGraph.

**Fale para fechar:**
- Recapitule os 4 requisitos obrigatórios atendidos (fine-tuning,
  LangChain, segurança/validação, organização do código) e onde encontrar
  cada entregável (relatório técnico em `reports/final_report.md`,
  código-fonte modularizado em `src/`, dataset em `data/`).
- Agradeça e encerre.

---

## Checklist de Requisitos a Citar em Voz Alta Durante o Vídeo

- [ ] Fine-tuning de LLM com dados médicos internos (protocolos, FAQs, laudos)
- [ ] Preprocessing, anonimização e curadoria dos dados
- [ ] Pipeline LangChain integrando a LLM customizada
- [ ] Consulta a base de dados estruturada (prontuários SQLite)
- [ ] Contextualização com dados atualizados do paciente
- [ ] Fluxo LangGraph: exames pendentes -> sugestão -> alertas
- [ ] Limites de atuação (nunca prescrever sem validação humana)
- [ ] Logging detalhado para auditoria
- [ ] Explainability (fonte da informação)
- [ ] Projeto modularizado + README completo
