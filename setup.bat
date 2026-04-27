@echo off
echo === UPI Intelligence Architecture Setup ===
echo.

echo [1/2] Installing Python dependencies...
pip install --upgrade pip
pip install "numpy>=2.1.0" --only-binary=:all:
if errorlevel 1 (
    echo ERROR: numpy install failed.
    pause
    exit /b 1
)
pip install streamlit langchain langchain-groq langchain-community langchain-huggingface chromadb pypdf sentence-transformers python-dotenv torch --only-binary=:all:
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Running ingestion - processing 127 circulars...
echo This takes 5-15 minutes on first run (downloads embedding model ~90MB, then embeds all PDFs).
echo.
python ingest.py
if errorlevel 1 (
    echo ERROR: Ingestion failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo === Setup complete! Launching UPI Intelligence Architecture... ===
streamlit run app.py
pause
