#!/usr/bin/env python3
"""
Simple test script to validate the utility functions (requires API keys).
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test if environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            if var == "SUPABASE_SERVICE_ROLE_KEY" and os.getenv("SUPABASE_ANON_KEY"):
                continue
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file")
        return False
    
    print("✅ All environment variables are set")
    return True

def test_imports():
    """Test if all imports work."""
    try:
        from utils.whisper_utils import transcribe_audio_bytes
        from utils.embed_utils import embed
        print("✅ All utility imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_embeddings():
    """Test embedding generation (requires API key)."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping embeddings test - no API key")
        return True
    
    try:
        from utils.embed_utils import embed
        test_text = "This is a test sentence for embedding generation."
        embedding = embed(test_text)
        
        if len(embedding) == 1536:
            print("✅ Embedding generation successful")
            return True
        else:
            print(f"❌ Unexpected embedding dimension: {len(embedding)}")
            return False
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Hyper Memory Tests...\n")
    
    tests = [
        ("Environment Variables", test_environment),
        ("Import Tests", test_imports),
        ("Embedding Test", test_embeddings),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}\n")
    
    print(f"🏁 Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check your configuration.")
        sys.exit(1)