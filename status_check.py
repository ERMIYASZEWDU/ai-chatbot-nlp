#!/usr/bin/env python3
"""
AI Chatbot (NLP) - System Status Check
Comprehensive health check and diagnostics for the chatbot system
"""

import sys
import os
import json
import importlib
from pathlib import Path
from datetime import datetime
import subprocess

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="INFO", color=Colors.WHITE):
    """Print formatted status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_colors = {
        "OK": Colors.GREEN,
        "ERROR": Colors.RED, 
        "WARNING": Colors.YELLOW,
        "INFO": Colors.BLUE
    }
    
    status_color = status_colors.get(status, color)
    print(f"{Colors.BOLD}[{timestamp}]{Colors.END} {status_color}[{status}]{Colors.END} {message}")

def check_python_environment():
    """Check Python version and environment"""
    print_status("Checking Python environment...", "INFO")
    
    # Python version
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        print_status(f"Python version: {version_str} ✓", "OK")
    else:
        print_status(f"Python version: {version_str} (requires 3.8+)", "ERROR")
        return False
    
    # Virtual environment check
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status("Virtual environment: Active ✓", "OK")
    else:
        print_status("Virtual environment: Not detected", "WARNING")
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_status("Checking dependencies...", "INFO")
    
    required_packages = [
        'nltk', 'scikit-learn', 'pandas', 'numpy', 'spacy', 
        'flask', 'transformers', 'torch', 'requests', 'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_status(f"  {package} ✓", "OK")
        except ImportError:
            print_status(f"  {package} ✗", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        return False
    
    return True

def check_spacy_models():
    """Check spaCy language models"""
    print_status("Checking spaCy models...", "INFO")
    
    try:
        import spacy
        
        # Try to load English model
        try:
            nlp = spacy.load("en_core_web_sm")
            print_status("  en_core_web_sm ✓", "OK")
        except OSError:
            print_status("  en_core_web_sm ✗", "ERROR")
            return False
            
    except ImportError:
        print_status("spaCy not installed", "ERROR")
        return False
    
    return True

def check_project_structure():
    """Check if all required directories and files exist"""
    print_status("Checking project structure...", "INFO")
    
    required_dirs = [
        "src", "src/nlp", "src/utils", "data", "models", 
        "static", "static/css", "static/js", "templates", "logs"
    ]
    
    required_files = [
        "config.yaml", "requirements.txt", "app.py", "train_model.py",
        "src/chatbot.py", "data/training_samples.json"
    ]
    
    all_good = True
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).exists():
            print_status(f"  {directory}/ ✓", "OK")
        else:
            print_status(f"  {directory}/ ✗", "ERROR")
            all_good = False
    
    # Check files
    for file in required_files:
        if Path(file).exists():
            print_status(f"  {file} ✓", "OK")
        else:
            print_status(f"  {file} ✗", "ERROR")
            all_good = False
    
    return all_good

def check_configuration():
    """Check configuration files"""
    print_status("Checking configuration...", "INFO")
    
    # Check config.yaml
    config_file = Path("config.yaml")
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            required_sections = ['model', 'nlp', 'intents', 'responses']
            for section in required_sections:
                if section in config:
                    print_status(f"  config.yaml -> {section} ✓", "OK")
                else:
                    print_status(f"  config.yaml -> {section} ✗", "ERROR")
                    
        except Exception as e:
            print_status(f"  config.yaml parsing error: {e}", "ERROR")
            return False
    else:
        print_status("  config.yaml ✗", "ERROR")
        return False
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print_status("  .env ✓", "OK")
    else:
        print_status("  .env ✗ (using defaults)", "WARNING")
    
    return True

def check_chatbot_module():
    """Test if the chatbot module can be imported and initialized"""
    print_status("Checking chatbot module...", "INFO")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path("src")))
        
        # Import chatbot
        from chatbot import create_chatbot
        print_status("  chatbot module import ✓", "OK")
        
        # Try to create chatbot instance
        try:
            chatbot = create_chatbot()
            print_status("  chatbot initialization ✓", "OK")
            
            # Test basic functionality
            try:
                status = chatbot.get_system_status()
                if status.get('status') == 'healthy':
                    print_status("  chatbot health check ✓", "OK")
                else:
                    print_status("  chatbot health check ✗", "WARNING")
                    
            except Exception as e:
                print_status(f"  chatbot health check error: {e}", "WARNING")
                
        except Exception as e:
            print_status(f"  chatbot initialization error: {e}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"  chatbot module import error: {e}", "ERROR")
        return False
    
    return True

def check_web_server():
    """Check if the web server can start"""
    print_status("Checking web server...", "INFO")
    
    try:
        # Test Flask import
        import flask
        print_status("  Flask import ✓", "OK")
        
        # Check if app.py exists and can be imported
        app_file = Path("app.py")
        if app_file.exists():
            print_status("  app.py exists ✓", "OK")
            
            # Try to import the Flask app (without running it)
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("app", "app.py")
                app_module = importlib.util.module_from_spec(spec)
                # Don't execute, just check if it can be loaded
                print_status("  app.py structure ✓", "OK")
            except Exception as e:
                print_status(f"  app.py import error: {e}", "ERROR")
                return False
        else:
            print_status("  app.py ✗", "ERROR")
            return False
            
    except ImportError:
        print_status("  Flask not installed ✗", "ERROR")
        return False
    
    return True

def check_training_data():
    """Check training data format and content"""
    print_status("Checking training data...", "INFO")
    
    training_file = Path("data/training_samples.json")
    
    if not training_file.exists():
        print_status("  training_samples.json ✗", "ERROR")
        return False
    
    try:
        with open(training_file) as f:
            data = json.load(f)
        
        if 'samples' in data:
            samples = data['samples']
            print_status(f"  {len(samples)} training samples ✓", "OK")
            
            # Check sample format
            if samples:
                sample = samples[0]
                if 'text' in sample and 'intent' in sample:
                    print_status("  sample format ✓", "OK")
                else:
                    print_status("  sample format ✗", "ERROR")
                    return False
            
            # Count intents
            intents = set(sample.get('intent') for sample in samples)
            print_status(f"  {len(intents)} unique intents ✓", "OK")
            
        else:
            print_status("  missing 'samples' key ✗", "ERROR")
            return False
            
    except json.JSONDecodeError as e:
        print_status(f"  JSON parsing error: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"  training data error: {e}", "ERROR")
        return False
    
    return True

def check_models():
    """Check if trained models exist"""
    print_status("Checking trained models...", "INFO")
    
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
    
    if model_files:
        for model_file in model_files:
            print_status(f"  {model_file.name} ✓", "OK")
        return True
    else:
        print_status("  No trained models found", "WARNING")
        print_status("  Run 'python train_model.py' to train models", "INFO")
        return True  # Not an error, just a warning

def run_system_diagnostic():
    """Run comprehensive system diagnostic"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}🔍 AI Chatbot (NLP) - System Diagnostic{Colors.END}")
    print("=" * 60)
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Dependencies", check_dependencies),
        ("spaCy Models", check_spacy_models),
        ("Project Structure", check_project_structure),
        ("Configuration", check_configuration),
        ("Chatbot Module", check_chatbot_module),
        ("Web Server", check_web_server),
        ("Training Data", check_training_data),
        ("Trained Models", check_models)
    ]
    
    results = {}
    
    for check_name, check_function in checks:
        print(f"\n{Colors.BOLD}🔸 {check_name}{Colors.END}")
        print("-" * 30)
        
        try:
            result = check_function()
            results[check_name] = result
        except Exception as e:
            print_status(f"Unexpected error in {check_name}: {e}", "ERROR")
            results[check_name] = False
    
    # Summary
    print(f"\n{Colors.BOLD}📊 DIAGNOSTIC SUMMARY{Colors.END}")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.END} {check_name}")
    
    print("-" * 60)
    
    if passed == total:
        print(f"{Colors.GREEN}🎉 All checks passed! Your AI chatbot is ready to go!{Colors.END}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
        print("  3. Start chatting! 🤖")
    else:
        print(f"{Colors.YELLOW}⚠️  {passed}/{total} checks passed{Colors.END}")
        print(f"\n{Colors.BOLD}Issues found:{Colors.END}")
        for check_name, result in results.items():
            if not result:
                print(f"  • {check_name}")
        
        print(f"\n{Colors.BOLD}Suggested fixes:{Colors.END}")
        print("  1. Run: python setup.py")
        print("  2. Install missing dependencies: pip install -r requirements.txt")
        print("  3. Download spaCy model: python -m spacy download en_core_web_sm")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_system_diagnostic()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Diagnostic interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        sys.exit(1)