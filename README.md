# 🔨 Sulfuras: Chatbot Inteligente com Contexto

Sulfuras é um chatbot inteligente desenvolvido para compreender documentos diversos (PDF, DOCX e CSV), permitindo que usuários interajam diretamente com os conteúdos enviados, gerando respostas contextuais inteligentes. Esse projeto foi realizado como Trabalho de Conclusão de Curso (TCC), com foco acadêmico e prático em inteligência artificial aplicada.

## 🎓 Autores

- Filipe S.
- Rafael C.
- Tatiana H.
- Hermes W.
- Vinicius M.

**Orientador:** M.e Weslley Rodrigues

---

## 📚 Introdução

Este projeto utiliza tecnologias avançadas de NLP (Natural Language Processing), armazenamento vetorial e modelos generativos para criar uma experiência interativa onde o chatbot "lembra" do contexto histórico e responde perguntas baseadas diretamente no conteúdo dos documentos enviados.

---

## 🛠️ Arquitetura do Projeto

<img width="1536" height="1024" alt="ChatGPT Image 26 de jun  de 2025, 20_00_28" src="https://github.com/user-attachments/assets/c3629de7-8c72-4463-b859-af8c6cb448e0" />

O projeto é estruturado da seguinte maneira:

### Interface do Usuário:
- **Streamlit:** Utilizado para criação de interfaces interativas e intuitivas diretamente no navegador.

### Modelo de Linguagem (LLM):
- **Groq (Modelo Llama3-8B):** Fornece respostas naturais, contextuais e inteligentes, hospedadas remotamente via API.

### Embeddings e Armazenamento Vetorial:
- **Sentence-Transformers (all-MiniLM-L6-v2):** Gera representações vetoriais de documentos e consultas.
- **ChromaDB:** Banco de dados vetorial para armazenamento, recuperação e busca eficiente dos documentos processados.

---

## 🚩 Requisitos para Execução

O projeto requer Python 3.10 ou superior. Instale as dependências com:

```bash
pip install -r requirements.txt
```

Exemplo de `requirements.txt`:

```text
streamlit
groq
sentence-transformers
chromadb
pandas
plotly
pymupdf
python-docx
pysqlite3-binary
```

---

## ⚙️ Como Utilizar

### Executando Localmente

Clone o repositório:

```bash
git clone "esserep"
cd Uther
```

Execute o aplicativo com Streamlit:

```bash
streamlit run app.py
```

Abra o navegador no endereço gerado pelo Streamlit (normalmente `http://localhost:8501`).

### Hospedagem Remota

O projeto pode ser hospedado facilmente em serviços como:

- [Render](https://render.com)
- [Railway](https://railway.app)

Adicione a variável de ambiente:

```bash
GROQ_API_KEY=<sua-chave-api>
```
## 📖 Licença de Estudo

Este projeto é licenciado sob a licença Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.pt_BR)).

Você tem liberdade para:
- Compartilhar: estudar e analisar o material em qualquer meio ou formato.
- Adaptar: remixar, transformar e criar a partir deste material.

Sob as condições seguintes:
- Atribuição: Você deve dar crédito adequado aos autores e indicar se mudanças foram feitas.
- Não Comercial: Você não pode usar o material para fins comerciais.
- CompartilhaIgual: Se você remixar, transformar ou criar a partir deste material, deve distribuir suas contribuições sob a mesma licença que o original.

---

## 📌 Estrutura de Diretórios

```bash
Uther/
├── assets/             # Imagens e outros arquivos estáticos
├── chromadb/           # Arquivos do banco vetorial
├── app.py              # Código principal da aplicação
├── requirements.txt    # Dependências Python
└── README.md           # Documentação principal
```

---

## 📞 Contato

Para quaisquer dúvidas ou sugestões, por favor entre em contato diretamente com os autores através do GitHub deste projeto.

---
