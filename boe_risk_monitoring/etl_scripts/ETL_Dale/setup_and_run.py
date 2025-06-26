import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_api_key():
    """Check if Pinecone API key is set"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not found!")
        print("\nTo set your API key:")
        print("Windows: set PINECONE_API_KEY=your_api_key_here")
        print("Linux/Mac: export PINECONE_API_KEY=your_api_key_here")
        print("\nOr add it to your .env file")
        return False
    print("✅ Pinecone API key found!")
    return True

def main():
    print("🚀 Setting up Pinecone upload environment...")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check API key
    if not check_api_key():
        return
    
    print("\n🎯 Ready to run! Execute: python pinecone_upload.py")

if __name__ == "__main__":
    main()