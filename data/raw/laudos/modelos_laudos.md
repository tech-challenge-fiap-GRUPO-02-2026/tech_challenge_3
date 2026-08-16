# Modelos Internos de Laudos, Receitas e Procedimentos

> Estes modelos servem como referência de **formato e linguagem** para o
> fine-tuning do assistente. Os dados de identificação são fictícios.

---

## Modelo 1 — Laudo de Eletrocardiograma

```
LAUDO DE ELETROCARDIOGRAMA
Paciente: {{NOME}}   Idade: {{IDADE}}   Data: {{DATA}}
Ritmo: sinusal, FC {{FC}} bpm.
Eixo elétrico: normal.
Intervalos: PR {{PR}}ms, QRS {{QRS}}ms, QTc {{QTC}}ms.
Alterações de repolarização: {{ALTERACOES}}.
Conclusão: {{CONCLUSAO}}.
Médico responsável: {{MEDICO}} — CRM {{CRM}}.
```

## Modelo 2 — Receita Padrão (uso hospitalar)

```
RECEITUÁRIO HOSPITALAR
Paciente: {{NOME}}   Leito: {{LEITO}}
1) {{MEDICAMENTO}} {{DOSE}} — via {{VIA}} — {{FREQUENCIA}}
2) ...
Observações: {{OBSERVACOES}}
Médico prescritor: {{MEDICO}} — CRM {{CRM}}
Assinatura e carimbo obrigatórios. Prescrição sujeita a validação da
farmácia clínica.
```

## Modelo 3 — Nota de Evolução Clínica

```
EVOLUÇÃO CLÍNICA — {{DATA}} {{HORA}}
Paciente: {{NOME}}   Registro: {{REGISTRO}}
Subjetivo: {{QUEIXA}}
Objetivo: PA {{PA}}, FC {{FC}}, FR {{FR}}, Sat O2 {{SATO2}}, Temp {{TEMP}}
Avaliação: {{AVALIACAO}}
Plano: {{PLANO}}
Médico: {{MEDICO}} — CRM {{CRM}}
```

## Modelo 4 — Solicitação de Exame

```
SOLICITAÇÃO DE EXAME
Paciente: {{NOME}}   Registro: {{REGISTRO}}
Exame solicitado: {{EXAME}}
Hipótese diagnóstica: {{HIPOTESE}}
Prioridade: {{PRIORIDADE}}
Médico solicitante: {{MEDICO}} — CRM {{CRM}}
```

---

## Diretriz de Uso pelo Assistente Virtual

O assistente pode **auxiliar na redação** desses documentos (preencher
estrutura, sugerir texto com base em protocolos), mas **nunca finaliza ou
assina** um laudo, receita ou solicitação sem revisão e validação de um
médico humano. Toda saída do assistente relacionada a esses modelos deve
ser marcada como "rascunho — requer validação médica".
