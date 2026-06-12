# Classificador de Vicio em Redes Sociais

Aplicacao Gradio para classificar o `Addicted_Score` usando um modelo ja treinado e salvo em `models/addicted_score_svm.joblib`.

## Como executar

```bash
pip install -r requirements.txt
python app.py
```

A aplicacao sobe por padrao em `http://127.0.0.1:7860`.

## Como funciona

- O app nao carrega dataset CSV em runtime.
- O formulario coleta as variaveis do problema.
- A predicao envia ao modelo somente as features que ele recebeu no treinamento salvo.
- `Student_ID` nao fica no formulario.
- `Affects_Academic_Performance` fica no formulario, mas nao entra no `predict` porque o modelo salvo foi treinado sem essa coluna.
- `Addicted_Score` e a saida prevista pelo modelo.
- O modelo e carregado uma vez com cache em memoria; cada envio do formulario apenas chama `predict`.

## Deploy

Para Hugging Face Spaces, use SDK `gradio` e mantenha `app.py`, `requirements.txt`, `src/` e `models/` no projeto.

Para Azure App Service ou Container Apps, configure o comando de start:

```bash
python app.py
```

O app respeita as variaveis de ambiente `PORT` e `HOST`. Em deploy, use `HOST=0.0.0.0` quando a plataforma nao definir automaticamente um marcador de Azure ou Hugging Face.
