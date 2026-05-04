"""
Quick Gemini API Diagnostic Script
Tests API connectivity, model availability, and basic functionality
"""

import asyncio
import os
import sys
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Colors for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test(msg):
    print(f"\n{BLUE}🧪 TEST: {msg}{RESET}")


def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")


def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")


async def test_api_key():
    """Test if API key is configured"""
    print_test("Checking API key configuration")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print_error("GEMINI_API_KEY not found in environment")
        return False

    print_success("API key found")
    genai.configure(api_key=api_key)
    return True


async def test_list_models():
    """List available models"""
    print_test("Listing available Gemini models")

    try:
        models = genai.list_models()
        available_models = []

        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"  • {model.name}")

        print_success(f"Found {len(available_models)} models with generateContent support")
        return available_models

    except Exception as e:
        print_error(f"Failed to list models: {e}")
        return []


async def test_simple_text_generation(model_name: str):
    """Test simple text generation"""
    print_test(f"Testing simple text generation with {model_name}")

    try:
        model = genai.GenerativeModel(model_name)

        # Test sync version first
        print("  Testing sync generate_content...")
        response = model.generate_content("Say 'Hello, World!' in Hebrew")

        if response.text:
            print_success(f"Sync generation works: {response.text[:50]}")
        else:
            print_error("Sync generation returned no text")
            return False

        # Test async version
        print("  Testing async generate_content_async...")
        response = await model.generate_content_async("Count to 3 in Hebrew")

        if response.text:
            print_success(f"Async generation works: {response.text[:50]}")
            return True
        else:
            print_error("Async generation returned no text")
            return False

    except Exception as e:
        print_error(f"Text generation failed: {type(e).__name__}: {str(e)[:200]}")
        return False


async def test_image_upload_and_generation(model_name: str, image_path: str = None):
    """Test image upload and vision capabilities"""
    print_test(f"Testing image upload and vision with {model_name}")

    if not image_path:
        print_warning("No image path provided, skipping image test")
        return None

    if not Path(image_path).exists():
        print_error(f"Image not found: {image_path}")
        return False

    try:
        # Upload file
        print(f"  Uploading {image_path}...")
        uploaded_file = genai.upload_file(image_path)
        print_success(f"Upload successful: {uploaded_file.name}")

        # Wait for processing
        await asyncio.sleep(2)

        # Try simple vision task
        model = genai.GenerativeModel(model_name)

        print("  Testing vision with short timeout (30s)...")
        try:
            response = await asyncio.wait_for(
                model.generate_content_async([
                    "Describe this image in one sentence.",
                    uploaded_file
                ]),
                timeout=30
            )

            if response.text:
                print_success(f"Vision works: {response.text[:100]}")
                return True
            else:
                print_error("Vision returned no text")
                return False

        except asyncio.TimeoutError:
            print_error("Vision timed out after 30s")
            return False

    except Exception as e:
        print_error(f"Image test failed: {type(e).__name__}: {str(e)[:200]}")
        return False


async def test_specific_models():
    """Test the specific models we're trying to use"""
    print_test("Testing specific model configurations")

    models_to_test = [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-002",
    ]

    working_models = []

    for model_name in models_to_test:
        print(f"\n  Testing {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async("Say OK")

            if response.text:
                print_success(f"{model_name} WORKS")
                working_models.append(model_name)
            else:
                print_warning(f"{model_name} returned no text")

        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "invalid" in error_msg.lower():
                print_error(f"{model_name} NOT FOUND/INVALID")
            else:
                print_error(f"{model_name} error: {error_msg[:100]}")

    return working_models


async def main():
    """Run all diagnostic tests"""

    print("=" * 80)
    print(f"{BLUE}🔧 GEMINI API DIAGNOSTIC TOOL{RESET}")
    print("=" * 80)

    # Test 1: API Key
    if not await test_api_key():
        print("\n" + "=" * 80)
        print_error("FATAL: Cannot proceed without API key")
        return

    # Test 2: List available models
    available_models = await test_list_models()

    # Test 3: Test specific models we care about
    working_models = await test_specific_models()

    # Test 4: Test Flash (should work)
    flash_works = False
    if "gemini-3-flash-preview" in working_models or "gemini-2.0-flash-exp" in working_models:
        flash_model = "gemini-3-flash-preview" if "gemini-3-flash-preview" in working_models else "gemini-2.0-flash-exp"
        flash_works = await test_simple_text_generation(flash_model)

    # Test 5: Test Pro (the problematic one)
    pro_works = False
    pro_model = None
    for model in ["gemini-3-pro-preview", "gemini-1.5-pro-002"]:
        if model in working_models:
            pro_model = model
            pro_works = await test_simple_text_generation(model)
            if pro_works:
                break

    # Test 6: Image test (optional)
    print("\n" + "=" * 80)
    image_path = input(f"\n{YELLOW}Enter path to test image (or press Enter to skip): {RESET}").strip()

    if image_path and Path(image_path).exists():
        if flash_works:
            flash_model = "gemini-3-flash-preview" if "gemini-3-flash-preview" in working_models else "gemini-2.0-flash-exp"
            await test_image_upload_and_generation(flash_model, image_path)

        if pro_works and pro_model:
            await test_image_upload_and_generation(pro_model, image_path)

    # Summary
    print("\n" + "=" * 80)
    print(f"{BLUE}📊 DIAGNOSTIC SUMMARY{RESET}")
    print("=" * 80)

    print(f"\nAPI Key: {GREEN}✓{RESET}")
    print(f"Available models: {len(available_models)}")
    print(f"Working models from our list: {len(working_models)}")

    if flash_works:
        print_success(f"Flash model works: {flash_model}")
    else:
        print_error("Flash model NOT working")

    if pro_works:
        print_success(f"Pro model works: {pro_model}")
    else:
        print_error("Pro model NOT working")

    # Recommendations
    print(f"\n{BLUE}💡 RECOMMENDATIONS:{RESET}")

    if not pro_works:
        print_error("\nPro model is not working. Possible issues:")
        print("  1. Model name is incorrect (Gemini 3 may not be released yet)")
        print("  2. Your API key doesn't have access to this model")
        print("  3. You've hit quota limits")
        print("  4. The model requires special access/approval")

        if working_models:
            print(f"\n  Try using one of these instead:")
            for model in working_models:
                print(f"    • {model}")

    if pro_works:
        print_success("\nPro model works for simple text!")
        print("  If it times out on images:")
        print("    1. Try smaller images (resize to ~2000x2000)")
        print("    2. Increase timeout to 10+ minutes")
        print("    3. Check your quota/rate limits in Google Cloud Console")
        print("    4. The model might just be very slow for complex images")


if __name__ == "__main__":
    asyncio.run(main())