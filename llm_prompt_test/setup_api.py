#!/usr/bin/env python3
"""
외부 API 설정 도구
빠른 설정을 위한 인터랙티브 스크립트
"""

import json
import os

def main():
    print("🚀 Fast External API Setup")
    print("=" * 40)
    print("This will enable super fast normalization using external APIs!")
    print()
    
    # 추천 순서로 API 소개
    apis_info = {
        "groq": {
            "name": "Groq",
            "description": "⚡ ULTRA FAST (recommended for speed)",
            "benefits": "- FREE tier with good limits\n  - Lightning fast inference\n  - Llama 3.1 70B model",
            "signup": "https://console.groq.com",
            "model": "llama-3.1-70b-versatile"
        },
        "openai": {
            "name": "OpenAI",
            "description": "🔥 Most reliable and widely used",
            "benefits": "- Very stable API\n  - GPT-4o-mini is cost effective\n  - Great quality",
            "signup": "https://platform.openai.com",
            "model": "gpt-4o-mini"
        },
        "claude": {
            "name": "Claude",
            "description": "🧠 High quality responses",
            "benefits": "- Excellent reasoning\n  - Haiku model is fast\n  - Good at following instructions",
            "signup": "https://console.anthropic.com",
            "model": "claude-3-5-haiku-20241022"
        },
        "gemini": {
            "name": "Google Gemini",
            "description": "💎 Good balance of speed and quality",
            "benefits": "- FREE tier available\n  - Good performance\n  - Easy to setup",
            "signup": "https://makersuite.google.com",
            "model": "gemini-1.5-flash"
        }
    }
    
    config = {
        "preferred_apis": ["groq", "openai", "claude", "gemini"],
        "groq": {"api_key": "", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-70b-versatile", "enabled": False},
        "openai": {"api_key": "", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "enabled": False},
        "claude": {"api_key": "", "base_url": "https://api.anthropic.com", "model": "claude-3-5-haiku-20241022", "enabled": False},
        "gemini": {"api_key": "", "base_url": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-1.5-flash", "enabled": False}
    }
    
    print("📋 Available APIs (in recommended order):")
    print()
    
    for i, (api_key, info) in enumerate(apis_info.items(), 1):
        print(f"{i}. {info['name']} - {info['description']}")
        print(f"   {info['benefits']}")
        print(f"   Signup: {info['signup']}")
        print()
    
    # Quick setup options
    print("⚡ Quick Setup Options:")
    print("1. Setup Groq only (FASTEST, recommended)")
    print("2. Setup OpenAI only (most reliable)")
    print("3. Setup multiple APIs (best redundancy)")
    print("4. Custom setup")
    print()
    
    while True:
        choice = input("Choose option (1-4): ").strip()
        
        if choice == "1":
            # Groq only
            setup_single_api("groq", apis_info["groq"], config)
            break
        elif choice == "2":
            # OpenAI only
            setup_single_api("openai", apis_info["openai"], config)
            break
        elif choice == "3":
            # Multiple APIs
            setup_multiple_apis(apis_info, config)
            break
        elif choice == "4":
            # Custom
            setup_custom_apis(apis_info, config)
            break
        else:
            print("Invalid choice. Please try again.")
    
    # Save config
    with open("api_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print()
    print("🎉 Setup complete!")
    print("✅ Configuration saved to api_config.json")
    
    # Show enabled APIs
    enabled = [name for name in ["groq", "openai", "claude", "gemini"] if config[name]["enabled"]]
    print(f"✅ Enabled APIs: {', '.join(enabled)}")
    
    print()
    print("🚀 Now you can run the pipeline with super fast normalization:")
    print("   python run_automated_pipeline.py")
    print()
    print("⚡ Speed improvement: 5-50x faster than local models!")

def setup_single_api(api_key, api_info, config):
    """단일 API 설정"""
    print()
    print(f"🔧 Setting up {api_info['name']}")
    print(f"   {api_info['description']}")
    print(f"   Get your API key from: {api_info['signup']}")
    print()
    
    api_key_input = input(f"Enter your {api_info['name']} API key: ").strip()
    
    if api_key_input:
        config[api_key]["api_key"] = api_key_input
        config[api_key]["enabled"] = True
        print(f"✅ {api_info['name']} configured!")
    else:
        print(f"❌ No API key provided for {api_info['name']}")

def setup_multiple_apis(apis_info, config):
    """여러 API 설정"""
    print()
    print("🔧 Setting up multiple APIs for redundancy")
    print("Recommended: Groq (primary) + OpenAI (backup)")
    print()
    
    for api_key, api_info in apis_info.items():
        enable = input(f"Setup {api_info['name']}? (y/n): ").lower().strip() == 'y'
        
        if enable:
            api_key_input = input(f"Enter {api_info['name']} API key: ").strip()
            if api_key_input:
                config[api_key]["api_key"] = api_key_input
                config[api_key]["enabled"] = True
                print(f"✅ {api_info['name']} configured!")
            else:
                print(f"❌ Skipped {api_info['name']}")
        print()

def setup_custom_apis(apis_info, config):
    """커스텀 API 설정"""
    print()
    print("🔧 Custom API setup")
    print()
    
    for api_key, api_info in apis_info.items():
        print(f"📡 {api_info['name']} - {api_info['description']}")
        print(f"   {api_info['benefits']}")
        print(f"   Signup: {api_info['signup']}")
        
        enable = input(f"Enable {api_info['name']}? (y/n): ").lower().strip() == 'y'
        
        if enable:
            api_key_input = input(f"Enter API key: ").strip()
            if api_key_input:
                config[api_key]["api_key"] = api_key_input
                config[api_key]["enabled"] = True
                print(f"✅ {api_info['name']} configured!")
            else:
                print(f"❌ Skipped {api_info['name']}")
        print()

if __name__ == "__main__":
    main()