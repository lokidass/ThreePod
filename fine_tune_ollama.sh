#!/bin/bash
# fine_tune_ollama.sh
# Complete script to fine-tune and deploy with Ollama

echo "🚀 Starting Ollama Fine-tuning Pipeline"

# Step 1: Install Ollama if not installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Step 2: Prepare dataset
echo "📊 Preparing dataset..."
python prepare_ollama_dataset.py

# Step 3: Pull base model (choose your preferred model)
echo "📥 Pulling base model..."
ollama pull qwen2.5-coder:7b
# Or use smaller model: ollama pull qwen2.5-coder:1.5b

# Step 4: Create Modelfile
echo "📝 Creating Modelfile..."
python create_modelfile.py

# Step 5: Create the model
echo "🔨 Creating Morphix model..."
ollama create morphix-threejs -f Modelfile

# Step 6: Test the model
echo "🧪 Testing model..."
ollama run morphix-threejs '{"action":"add_sphere","radius":0.5,"position":[0,1,0],"color":"0xFF0000","label":"test"}'

echo "✅ Fine-tuning complete!"
echo "Run with: ollama run morphix-threejs"