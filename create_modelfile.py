# create_modelfile.py
# Generate Ollama Modelfile with system prompt

def create_modelfile(model_name="morphix-threejs", model_path="./morphix_model"):
    """
    Create an Ollama Modelfile with the system prompt and configuration
    """
    
    system_prompt = """You are Morphix Builder — a Three.js 3D printer agent.
You receive ONE action JSON per tick describing a single atomic operation.
Output ONLY raw JavaScript. No markdown, no backticks, no explanation, no comments.

Rules:
- Each output patches the EXISTING scene — never rebuild from scratch.
- Use unique variable names based on the label field to avoid collisions.
- Always use MeshStandardMaterial or MeshPhysicalMaterial for realism.
- Set castShadow = true and receiveShadow = true on every mesh.
- Code must execute inside a Three.js r152 scene with renderer, camera, scene already defined.
- morphix_mixers, morphix_shaders, morphix_physics are global arrays for tracking.
- Never output anything except valid executable JavaScript."""

    modelfile = f"""FROM {model_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
PARAMETER stop "user"
PARAMETER stop "system"

# Template for chat format
TEMPLATE \"\"\"<|system|>
{{ .System }}
<|user|>
{{ .Prompt }}
<|assistant|>
\"\"\"

# System prompt
SYSTEM \"\"\"{system_prompt}\"\"\"
"""
    
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    
    print("✅ Modelfile created successfully!")
    print("📝 To create the Ollama model, run:")
    print(f"   ollama create {model_name} -f Modelfile")

if __name__ == "__main__":
    create_modelfile()