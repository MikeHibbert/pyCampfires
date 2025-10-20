# Ollama Integration Demos

This directory contains demonstration scripts for using Ollama with the Campfires framework.

## Prerequisites

Before running these demos, ensure you have:

1. **Ollama installed and running**:
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   # Start Ollama server
   ollama serve
   ```

2. **Required models downloaded**:
   ```bash
   # For basic text generation
   ollama pull llama2
   
   # For multimodal capabilities (image analysis)
   ollama pull llava
   
   # Optional: Other models you want to test
   ollama pull codellama
   ollama pull mistral
   ```

3. **Python dependencies**:
   ```bash
   pip install aiohttp pillow
   ```

## Demo Scripts

### 1. Quick Ollama Test (`quick_ollama_test.py`)

A simple script to verify that Ollama integration is working correctly.

**What it tests:**
- Ollama server connection
- Model listing
- Basic text generation
- Chat functionality
- Factory integration

**Usage:**
```bash
python demos/quick_ollama_test.py
```

**Expected output:**
- Lists available models
- Generates a simple response
- Tests chat functionality
- Verifies factory integration

### 2. Comprehensive Ollama Demo (`ollama_demo.py`)

A full-featured demonstration of all Ollama capabilities.

**What it demonstrates:**
- Basic text generation with different parameters
- Interactive chat conversations
- Multimodal image analysis (requires llava model)
- Model management (listing, pulling)
- MCP protocol integration
- Error handling and best practices

**Usage:**
```bash
python demos/ollama_demo.py
```

**Features demonstrated:**
- Text generation with custom prompts
- Multi-turn conversations
- Image analysis and description
- Object detection in images
- Text extraction from images
- Model downloading and management

## Configuration

### Basic Configuration

```python
from campfires.core.ollama import OllamaConfig

config = OllamaConfig(
    base_url="http://localhost:11434",  # Default Ollama server
    model="llama2",                     # Model to use
    temperature=0.7,                    # Response creativity
    max_tokens=1000                     # Maximum response length
)
```

### Multimodal Configuration

```python
from campfires.core.multimodal_ollama import MultimodalOllamaConfig

config = MultimodalOllamaConfig(
    base_url="http://localhost:11434",
    text_model="llama2",               # For text generation
    vision_model="llava",              # For image analysis
    temperature=0.7,
    max_tokens=1000
)
```

### Factory Integration

```python
from campfires.core.factory import DynamicCamper

config = {
    'name': 'my_ollama_camper',
    'llm_provider': 'ollama',
    'ollama_base_url': 'http://localhost:11434',
    'ollama_model': 'llama2'
}

camper = DynamicCamper(config)
```

## Troubleshooting

### Common Issues

1. **"Connection refused" error**:
   - Make sure Ollama is running: `ollama serve`
   - Check if the server is accessible at `http://localhost:11434`

2. **"Model not found" error**:
   - Install the required model: `ollama pull llama2`
   - Check available models: `ollama list`

3. **"No models found" message**:
   - Download at least one model: `ollama pull llama2`
   - Wait for the download to complete

4. **Slow responses**:
   - Ollama performance depends on your hardware
   - Consider using smaller models for faster responses
   - Adjust `max_tokens` to limit response length

5. **Image analysis not working**:
   - Make sure you have the `llava` model: `ollama pull llava`
   - Verify image file exists and is in a supported format (PNG, JPEG, etc.)

### Performance Tips

1. **Model Selection**:
   - `llama2:7b` - Good balance of speed and quality
   - `llama2:13b` - Better quality, slower
   - `codellama` - Optimized for code generation
   - `mistral` - Fast and efficient

2. **Configuration Tuning**:
   - Lower `temperature` (0.1-0.3) for more focused responses
   - Higher `temperature` (0.7-1.0) for more creative responses
   - Adjust `max_tokens` based on your needs

3. **Hardware Considerations**:
   - Ollama uses GPU acceleration when available
   - More RAM allows for larger models
   - SSD storage improves model loading times

## Next Steps

After running these demos successfully:

1. **Integrate into your projects**: Use the patterns shown in these demos
2. **Explore other models**: Try different Ollama models for various tasks
3. **Customize configurations**: Adjust parameters for your specific use cases
4. **Build applications**: Use Ollama with the full Campfires framework

## Support

For issues specific to:
- **Ollama**: Visit [Ollama GitHub](https://github.com/jmorganca/ollama)
- **Campfires**: Check the main project documentation
- **Integration**: Review the demo code and error messages