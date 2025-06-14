"""
Hyper Memory - Audio Recording, Transcription, and Semantic Search App

A Streamlit application that records audio, transcribes it using OpenAI Whisper,
stores transcripts with embeddings in Supabase, and enables semantic search.
"""

import os
import tempfile
from typing import List, Optional, Tuple
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from supabase import create_client, Client
from dotenv import load_dotenv

# Import utility modules
from utils.whisper_utils import transcribe_audio_bytes
from utils.embed_utils import embed

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Hyper Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit menu and footer
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


@st.cache_resource
def init_supabase() -> Client:
    """Initialize Supabase client."""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            st.error("Missing Supabase configuration. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
            st.stop()
        
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {str(e)}")
        st.stop()


def validate_environment() -> bool:
    """Validate required environment variables."""
    required_vars = [
        "OPENAI_API_KEY",
        "SUPABASE_URL", 
        "SUPABASE_SERVICE_ROLE_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            # Check for alternative key names
            if var == "SUPABASE_SERVICE_ROLE_KEY" and os.getenv("SUPABASE_ANON_KEY"):
                continue
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please set these in your .env file or environment.")
        return False
    
    return True


def save_to_supabase(content: str, embedding: List[float], supabase: Client) -> bool:
    """Save content and embedding to Supabase."""
    try:
        result = supabase.table("docs").insert({
            "content": content,
            "embedding": embedding
        }).execute()
        
        return len(result.data) > 0
    except Exception as e:
        st.error(f"Failed to save to Supabase: {str(e)}")
        return False


def search_supabase(query_embedding: List[float], supabase: Client, limit: int = 5) -> List[Tuple[str, str, float]]:
    """Search for similar content in Supabase."""
    try:
        result = supabase.rpc("semantic_search", {
            "query_embedding": query_embedding,
            "match_count": limit
        }).execute()
        
        return [(row["id"], row["content"], row["distance"]) for row in result.data]
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []


def record_tab(supabase: Client):
    """Audio recording and transcription tab."""
    st.header("🎙️ Record Audio")
    st.write("Record audio, transcribe it, and save to your memory bank.")
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="🎙️ Start recording",
        recording_color="#ff6b6b",
        neutral_color="#fafafa",
        icon_name="microphone",
        icon_size="2x"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("Transcribing audio..."):
            try:
                # Transcribe audio
                transcript = transcribe_audio_bytes(audio_bytes)
                
                if transcript:
                    st.success("Transcription completed!")
                    
                    # Display transcript in expandable text area
                    with st.expander("📝 Transcript", expanded=True):
                        edited_transcript = st.text_area(
                            "Edit transcript if needed:",
                            value=transcript,
                            height=150,
                            help="You can edit the transcript before saving"
                        )
                    
                    # Save button
                    if st.button("💾 Save to Memory Bank", type="primary"):
                        with st.spinner("Generating embeddings and saving..."):
                            try:
                                # Generate embedding
                                embedding = embed(edited_transcript)
                                
                                # Save to Supabase
                                if save_to_supabase(edited_transcript, embedding, supabase):
                                    st.toast("✅ Saved to Supabase!", icon="✅")
                                    st.balloons()
                                else:
                                    st.error("Failed to save to database")
                                    
                            except Exception as e:
                                st.error(f"Error processing transcript: {str(e)}")
                else:
                    st.warning("No speech detected in the audio. Please try recording again.")
                    
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")


def search_tab(supabase: Client):
    """Semantic search tab."""
    st.header("🔍 Search Memories")
    st.write("Search through your recorded memories using semantic similarity.")
    
    # Search input
    query = st.text_input(
        "What are you looking for?",
        placeholder="Enter your search query...",
        help="Type your search query and press Enter"
    )
    
    # Search button or enter key
    if query:
        with st.spinner("Searching..."):
            try:
                # Generate embedding for query
                query_embedding = embed(query)
                
                # Search for similar content
                results = search_supabase(query_embedding, supabase)
                
                if results:
                    st.success(f"Found {len(results)} similar memories:")
                    
                    # Display results
                    for i, (doc_id, content, distance) in enumerate(results, 1):
                        similarity_score = 1 - distance  # Convert distance to similarity
                        
                        with st.expander(
                            f"📄 Memory {i} (Similarity: {similarity_score:.2%})", 
                            expanded=i <= 2  # Expand first 2 results
                        ):
                            st.write(content)
                            st.caption(f"Document ID: {doc_id} | Distance: {distance:.4f}")
                else:
                    st.info("🤷 No similar memories found. Try a different search query.")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")


def main():
    """Main application."""
    # Title and description
    st.title("🧠 Hyper Memory")
    st.markdown("*Record, transcribe, and search your audio memories with AI*")
    
    # Validate environment
    if not validate_environment():
        st.stop()
    
    # Initialize Supabase
    supabase = init_supabase()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎙️ Record", "🔍 Search"])
    
    with tab1:
        record_tab(supabase)
    
    with tab2:
        search_tab(supabase)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, OpenAI, and Supabase"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()