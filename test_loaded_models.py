#!/usr/bin/env python3
import subprocess
import json
import base64
import time

def test_model(model, prompt, width, height, steps, guidance, test_name):
    print(f"\nTesting {model} - {test_name} ({width}x{height})...")
    
    data = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "seed": 42,
        "negative_prompt": "ugly, blurry, low quality, distorted"
    }
    
    try:
        # Get response and parse it
        response = subprocess.run(
            ['curl', '-X', 'POST', 'http://localhost:8000/v1/images/generations',
             '-H', 'Content-Type: application/json',
             '-d', json.dumps(data)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if response.returncode == 0 and response.stdout:
            try:
                result = json.loads(response.stdout)
                if 'image' in result:
                    # Save image
                    img_data = base64.b64decode(result['image'])
                    filename = f"test_{model}_{test_name}.png"
                    with open(filename, 'wb') as f:
                        f.write(img_data)
                    print(f"✓ Success! Image saved to {filename}")
                    return True, filename
                else:
                    print(f"✗ Unexpected response format")
                    return False, None
            except json.JSONDecodeError:
                print(f"✗ Failed to parse JSON response")
                return False, None
        else:
            print(f"✗ Failed with return code {response.returncode}")
            if response.stderr:
                print(f"Error: {response.stderr}")
            return False, None
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 30 seconds")
        return False, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None

# Test prompts
long_prompt = """A steampunk inventor's workshop filled with intricate brass machinery and clockwork mechanisms. 
The elderly inventor, wearing vintage goggles pushed up on their forehead, carefully adjusts a complex mechanical bird 
with iridescent copper feathers. Sunlight streams through tall arched windows, catching the dust motes and steam from 
various experimental devices. The workshop walls are lined with blueprints, gears of all sizes, and shelves containing 
mysterious glowing vials. In the background, a massive orrery slowly rotates, its planets following precise mechanical paths."""

short_prompt = "A magical forest with glowing mushrooms and fireflies at twilight"

# Dictionary to collect results
results = {}

if __name__ == "__main__":
    # First restart the server with the default configuration
    print("Starting server with default configuration...")
    subprocess.run(['pkill', '-f', 'python.*server.py'], capture_output=True)
    time.sleep(2)
    
    # Start server in background
    server_proc = subprocess.Popen(['python3', 'server.py'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(20)
    
    # Test FLUX
    print("\n=== Testing FLUX.1 Schnell ===")
    success, filename = test_model("flux-schnell", short_prompt, 1024, 1024, 4, 0.0, "square_1024")
    results['flux-schnell'] = {'success': success, 'files': [filename] if success else []}
    
    # Test SDXL
    print("\n=== Testing Stable Diffusion XL ===")
    success, filename = test_model("sdxl", long_prompt, 1024, 1024, 30, 7.5, "square_1024_compel")
    results['sdxl'] = {'success': success, 'files': [filename] if success else []}
    
    # Test Playground
    print("\n=== Testing Playground v2.5 ===")
    success, filename = test_model("playground", long_prompt, 1024, 1024, 50, 3.0, "square_1024_compel")
    if success:
        results['playground'] = {'success': True, 'files': [filename]}
        # Test different aspect ratio
        success2, filename2 = test_model("playground", short_prompt, 1280, 768, 50, 3.0, "landscape_1280x768")
        if success2:
            results['playground']['files'].append(filename2)
    
    print("\n\n=== Test Summary ===")
    for model, result in results.items():
        if result['success']:
            print(f"✓ {model}: Success - Generated {len(result['files'])} images")
        else:
            print(f"✗ {model}: Failed")
    
    # Keep server running for analysis
    print("\nServer is still running. Press Ctrl+C to stop.")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        server_proc.terminate()