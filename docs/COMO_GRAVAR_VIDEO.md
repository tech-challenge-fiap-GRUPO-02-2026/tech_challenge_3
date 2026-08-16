# Como Gravar o Vídeo de Demonstração

Este guia é sobre **como gravar e publicar** o vídeo. Para saber **o que
falar e mostrar em cada minuto**, use `docs/video_script.md`.

## 1. Requisitos do vídeo (segundo o enunciado da Fase 3)

- Duração **máxima de 15 minutos**.
- Deve demonstrar:
  1. Treinamento e funcionamento da LLM personalizada;
  2. Execução de um fluxo automatizado (LangGraph);
  3. Resposta a perguntas clínicas contextualizadas (LangChain + RAG + prontuário);
  4. Logs e validação das respostas (auditoria e guardrails).

## 2. Ferramentas recomendadas para gravar

| Ferramenta | Plataforma | Observação |
|------------|------------|------------|
| **OBS Studio** (gratuito) | Windows/Mac/Linux | Mais controle, grava tela + webcam + áudio separados |
| **Gravador de Jogos do Windows** (`Win + G`) | Windows | Já vem instalado, simples para gravar uma janela |
| **Loom** | Navegador | Rápido, já gera link compartilhável |
| **Zoom** (gravar reunião só com você) | Qualquer | Útil se o grupo quiser gravar juntos remotamente |

### Configuração sugerida (OBS)
1. `Fontes` → `+` → `Captura de Janela` → selecione o terminal.
2. Adicione uma segunda fonte `Captura de Janela` para o navegador (para
   mostrar o `index.html` e o diagrama de arquitetura).
3. `Configurações` → `Saída` → resolução mínima **1920x1080**, 30fps.
4. Teste o áudio do microfone antes de gravar (`Configurações` → `Áudio`).

## 3. Passo a passo de gravação

1. **Ensaie uma vez sem gravar**, seguindo `docs/video_script.md`, para
   validar tempo e comandos.
2. Antes de gravar, rode os comandos de preparação da seção "Preparação"
   do roteiro (anonimização, dataset, fine-tuning se aplicável) — isso
   evita tempos de espera ao vivo.
3. Aumente a fonte do terminal para pelo menos 18pt (`Ctrl` + `+` na
   maioria dos terminais) para garantir legibilidade em qualquer tela.
4. Grave em blocos, se preferir (um bloco por seção do roteiro), e
   depois una os cortes na edição — mais seguro do que tentar gravar
   tudo em uma única tomada.
5. Sempre que rodar um comando, **pause 2-3 segundos** antes de falar
   sobre o resultado, para dar tempo de leitura na edição.

## 4. Edição (opcional, mas recomendado)

- **DaVinci Resolve** (gratuito) ou **CapCut** (simples, multiplataforma)
  para cortar silêncios, juntar blocos e adicionar legendas/títulos de
  seção (ex.: "Bloco 3 — Assistente LangChain").
- Adicione uma tela de abertura com o nome do grupo, disciplina
  (Tech Challenge Fase 3 — FIAP) e data.
- Verifique a duração final: **deve ser ≤ 15 minutos**.

## 5. Publicação

1. Suba o vídeo em um canal do **YouTube** (pode ser "não listado", não
   precisa ser público) — mesmo padrão usado na Fase 2.
2. Copie o link e adicione na seção **🎬 Vídeo Demonstração** do
   `README.md` deste projeto (substitua o placeholder).
3. Confirme que o link abre corretamente em uma aba anônima antes de
   entregar (garante que não está restrito por permissão de conta).

## 6. Checklist final antes de entregar

- [ ] Vídeo com no máximo 15 minutos.
- [ ] Os 4 blocos de conteúdo obrigatório aparecem claramente (LLM
      treinada, fluxo automatizado, perguntas contextualizadas, logs).
- [ ] Áudio limpo, sem ruído de fundo, voz audível do início ao fim.
- [ ] Terminal legível (fonte grande, sem elementos cortados na tela).
- [ ] Link do vídeo atualizado no `README.md`.
- [ ] Vídeo testado em aba anônima (acesso público/não-listado confirmado).
