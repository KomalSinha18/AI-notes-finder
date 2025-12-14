# 🚀 Quick Start - Enhanced Version

## Installation

1. **Install enhanced dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have a `.env` file:**
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Basic Usage

```bash
python app_enhanced.py
```

That's it! The enhanced version works just like the original, but with more features.

## Key Differences from Original

### ✅ **What's New:**

1. **Supports PDF, DOCX, Markdown** (not just TXT)
2. **Remembers conversation** (context-aware follow-ups)
3. **Streaming responses** (see answers as they generate)
4. **Query statistics** (type `stats` in CLI)
5. **Better error handling**
6. **Configuration file support**

### 🔄 **Same Interface:**

```python
from app_enhanced import EnhancedAINotesFinder

app = EnhancedAINotesFinder()
app.process_notes("./notes")
result = app.ask("What is machine learning?")
```

## New CLI Commands

When running `python app_enhanced.py`:

- **`stats`** - Show query statistics
- **`clear`** - Clear conversation history
- **`quit`** or **`exit`** - Exit

## Enable Hybrid Search

Edit `app_enhanced.py` in `main()`:

```python
app = EnhancedAINotesFinder(
    use_hybrid_search=True  # Enable hybrid search
)
```

## See Full Documentation

Check `ENHANCEMENTS.md` for detailed documentation of all features!

