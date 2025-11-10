# DeepSeek‑OCR API (FastAPI)

API HTTP em Python (FastAPI) para processar **imagens** e **PDFs** com o **DeepSeek‑OCR** rodando localmente (GPU).

A API usa `transformers` e o método `model.infer()` do DeepSeek‑OCR para extrair texto e converter documentos em Markdown.

> Requisitos (mínimo): CUDA 11.8, PyTorch 2.6.0, Flash‑Attention 2.7.3, GPU NVIDIA (A100/RTX, etc.).

## 1) Pré‑requisitos

- Drivers NVIDIA + CUDA 11.8 (host)
- `nvidia-container-toolkit` instalado e configurado
- Docker + Docker Compose

## 2) Subir tudo com Docker

```bash
git clone <SEU-REPO>
cd <SEU-REPO>

# Build & up
docker compose up --build
```

A API FastAPI estará em `http://localhost:3000`.

## 3) Exemplos de uso

### Imagem (PNG/JPG)

```bash
curl -X POST http://localhost:3000/ocr/image \
  -F file=@nota_fiscal.jpg \
  -F prompt='<image>\n<|grounding|>Convert the document to markdown.'
```

### PDF

```bash
curl -X POST http://localhost:3000/ocr/pdf \
  -F file=@contrato.pdf
```

## 4) Estrutura do projeto

```
.
├─ inference/                  # API FastAPI com DeepSeek‑OCR
│  ├─ main.py                  # FastAPI com DeepSeek‑OCR (Transformers)
│  ├─ requirements.txt         # dependências Python
│  ├─ Dockerfile               # imagem Docker
│  ├─ install-deps.ps1         # script para instalar deps no Windows
│  └─ start-server.ps1         # script para iniciar servidor no Windows
├─ examples/                   # arquivos de exemplo para teste
├─ docker-compose.yml          # orquestra a API
└─ README.md                   # como rodar
```

## 5) Notas importantes

* Ajuste `image_size`/`base_size` e o **prompt** conforme o tipo de documento (tabelas, figuras, etc.).
* O modelo é exigente em VRAM; ajuste os parâmetros de inferência se necessário.
* A API processa PDFs página a página, convertendo cada página em imagem antes do OCR.

## 6) Dicas de prompt

- Documento para Markdown:
  ```
  <image>
  <|grounding|>Convert the document to markdown.
  ```

- OCR simples (sem layout): `"<image>\nFree OCR."`
- Foco em figuras: `"<image>\nParse the figure."`
- Localização: `"<image>\nLocate <|ref|>IBAN<|/ref|> in the image."`

## 7) Segurança & limites

* Esta API não persiste arquivos por padrão (usa diretório temporário).
* Adicione autenticação (ex.: API key via header) antes de expor fora da rede local.
* Para produção, considere: logs estruturados, limitação de tamanho, fila de jobs e retries para PDFs longos.

## 8) Desenvolvimento local (sem Docker)

### Windows (PowerShell)

```powershell
cd inference
# Instalar dependências (primeira vez)
.\install-deps.ps1
# Iniciar servidor
.\start-server.ps1
```

### Linux/Mac

```bash
cd inference
pip install -r requirements.txt
python main.py
```

A API estará disponível em `http://localhost:3000` por padrão.

## 9) Documentação Swagger/OpenAPI

A API possui documentação interativa completa gerada automaticamente pelo FastAPI.

### Acessar a documentação

Com o servidor rodando, acesse:

- **Swagger UI (Interativo):** http://localhost:3000/docs
- **ReDoc (Visualização alternativa):** http://localhost:3000/redoc
- **OpenAPI JSON:** http://localhost:3000/openapi.json

### Recursos da documentação

- ✨ **Teste interativo** - Execute requisições diretamente do navegador
- 📋 **Esquemas completos** - Visualize modelos de requisição/resposta com exemplos
- 🎯 **Exemplos de uso** - Múltiplos exemplos para diferentes casos de uso
- 📖 **Descrições detalhadas** - Documentação completa de todos os endpoints

### Endpoints disponíveis

#### Health Check
- `GET /health` - Verificação básica de saúde da API
- `GET /health/detailed` - Informações detalhadas do sistema e modelo

#### OCR
- `POST /ocr/image` - Extração de texto de imagens (PNG, JPG, WEBP, BMP, TIFF)
- `POST /ocr/pdf` - Extração de texto de PDFs multi-página

Para exemplos detalhados e informações completas, consulte:
- **Documentação Interativa:** http://localhost:3000/docs
- **Guia de API:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## Licença

Este projeto é fornecido como está, sem garantias. Use por sua conta e risco.

