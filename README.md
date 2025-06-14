# 🧠 Hyper Memory

A powerful Streamlit application that records audio, transcribes it using OpenAI Whisper, stores transcripts with embeddings in Supabase, and enables semantic search through your audio memories.

## ✨ Features

- 🎙️ **Audio Recording**: Record audio directly in the browser
- 🗣️ **Speech-to-Text**: Transcribe audio using OpenAI Whisper
- 🔍 **Semantic Search**: Find memories using natural language queries
- 📊 **Vector Storage**: Store embeddings in Supabase with pgvector
- 🎨 **Clean UI**: Beautiful Streamlit interface with dark theme

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd hyper-memory
pip install -r requirements.txt
```

### 2. Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# OpenAI API Key for Whisper and embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Alternative: Use anon key if RLS is disabled
# SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🗄️ Supabase Setup

### 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Wait for the database to be ready

### 2. Enable pgvector Extension

In your Supabase SQL Editor, run:

```sql
create extension if not exists pgvector;
```

### 3. Create Database Schema

Run the SQL from `db/schema.sql` in your Supabase SQL Editor:

```sql
-- Enable pgvector extension for vector similarity search
create extension if not exists pgvector;

-- Create docs table to store transcriptions and their embeddings
create table if not exists docs (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(1536) not null,
  created_at timestamp with time zone default now()
);

-- Create index for faster vector similarity search
create index if not exists docs_embedding_idx on docs 
using hnsw (embedding vector_cosine_ops);

-- Function to perform semantic search using cosine similarity
create or replace function semantic_search(
  query_embedding vector(1536),
  match_count int default 5
)
returns table(id uuid, content text, distance float)
language sql stable as $$
  select 
    docs.id, 
    docs.content, 
    docs.embedding <-> query_embedding as distance
  from docs
  order by docs.embedding <-> query_embedding
  limit match_count;
$$;
```

### 4. Get API Keys

1. Go to Project Settings > API
2. Copy your `Project URL` (SUPABASE_URL)
3. Copy your `service_role` key (SUPABASE_SERVICE_ROLE_KEY)

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your environment variables in the Streamlit Cloud secrets
5. Deploy!

### Railway Deployment

1. Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Deploy to Railway:
   - Connect your GitHub repo
   - Add environment variables
   - Deploy automatically

### Alternative: Render

Create a `render.yaml`:

```yaml
services:
  - type: web
    name: hyper-memory
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_ROLE_KEY
        sync: false
```

## 🔧 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key for Whisper and embeddings | ✅ Yes |
| `SUPABASE_URL` | Your Supabase project URL | ✅ Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (full access) | ✅ Yes* |
| `SUPABASE_ANON_KEY` | Alternative: Supabase anon key (if RLS disabled) | 🔄 Alternative |

*Use `SUPABASE_SERVICE_ROLE_KEY` for full access or `SUPABASE_ANON_KEY` if you've disabled Row Level Security.

## 🏗️ Project Structure

```
hyper-memory/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── utils/
│   ├── __init__.py        # Package initialization
│   ├── whisper_utils.py   # OpenAI Whisper transcription
│   └── embed_utils.py     # OpenAI embeddings generation
├── db/
│   └── schema.sql         # Database schema and functions
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── README.md              # This file
```

## 🎯 Usage

### Recording Audio

1. Click the **🎙️ Record** tab
2. Click **"🎙️ Start recording"** to begin
3. Click **"■ Stop"** when finished
4. Review and edit the transcript if needed
5. Click **"💾 Save to Memory Bank"** to store

### Searching Memories

1. Click the **🔍 Search** tab
2. Enter your search query in natural language
3. Press Enter to search
4. Browse through similar memories ranked by relevance

## 🔧 Development

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for modules and functions
- Handle exceptions with user-friendly error messages

### Testing

```bash
# Install the app locally
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Test with sample audio recordings
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Missing environment variables"**
   - Ensure all required variables are set in your `.env` file
   - Check for typos in variable names

2. **"Failed to initialize Supabase"**
   - Verify your Supabase URL and API key
   - Ensure your Supabase project is running

3. **"OpenAI API error"**
   - Check your OpenAI API key
   - Ensure you have sufficient API credits

4. **"Audio file too large"**
   - OpenAI Whisper has a 25MB file size limit
   - Record shorter audio clips

### Performance Tips

- For long audio recordings, consider chunking them into smaller segments
- Use the service role key for better performance (bypasses RLS)
- Enable caching for frequent searches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the amazing web app framework
- [OpenAI](https://openai.com) for Whisper and embeddings API
- [Supabase](https://supabase.com) for the backend and vector database
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search