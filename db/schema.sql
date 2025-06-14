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