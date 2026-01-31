"""
Test script for MUST++ Frontend API

Comprehensive test suite including:
- Health check
- Script detection
- Analysis with various languages
- Response structure validation
- Edge cases and error handling
- Rate limiting (optional)
- Cache behavior

Run this to verify all endpoints work correctly.
"""

import requests
import json
import time
import sys
from typing import List, Tuple, Optional

BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 5

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{color}{text}{Colors.RESET}"


def print_header(title: str):
    """Print a section header."""
    print()
    print(colored("=" * 60, Colors.BLUE))
    print(colored(f" {title}", Colors.BOLD))
    print(colored("=" * 60, Colors.BLUE))


def print_result(passed: bool, message: str):
    """Print a test result."""
    if passed:
        print(f"  {colored('✓', Colors.GREEN)} {message}")
    else:
        print(f"  {colored('✗', Colors.RED)} {message}")


def test_health() -> bool:
    """Test health endpoint."""
    print_header("Health Check")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        data = r.json()
        
        print_result(r.status_code == 200, f"Status code: {r.status_code}")
        print_result(data.get('status') == 'healthy', f"Status: {data.get('status')}")
        print_result('version' in data, f"Version: {data.get('version')}")
        print_result('stats' in data, f"Stats available: {'stats' in data}")
        
        return r.status_code == 200 and data.get('status') == 'healthy'
    except Exception as e:
        print_result(False, f"ERROR: {e}")
        return False


def test_stats() -> bool:
    """Test stats endpoint."""
    print_header("Statistics Endpoint")
    try:
        r = requests.get(f"{BASE_URL}/stats", timeout=TIMEOUT)
        data = r.json()
        
        print_result(r.status_code == 200, f"Status code: {r.status_code}")
        print_result('total_requests' in data, f"Total requests: {data.get('total_requests')}")
        print_result('cache_hit_rate' in data, f"Cache hit rate: {data.get('cache_hit_rate')}")
        print_result('uptime_seconds' in data, f"Uptime: {data.get('uptime_seconds')}s")
        
        return r.status_code == 200
    except Exception as e:
        print_result(False, f"ERROR: {e}")
        return False


def test_detect_script() -> bool:
    """Test script detection endpoint."""
    print_header("Script Detection")
    
    test_cases = [
        ("Hello world", "latin"),
        ("नमस्ते दुनिया", "devanagari"),
        ("வணக்கம் உலகம்", "tamil"),
        ("Hello नमस्ते", None),  # Mixed
    ]
    
    all_passed = True
    for text, expected_primary in test_cases:
        try:
            r = requests.post(f"{BASE_URL}/detect-script", 
                            json={"text": text}, 
                            timeout=TIMEOUT)
            result = r.json()
            actual = result.get('primary_script', 'unknown')
            is_mixed = result.get('is_mixed', False)
            
            if expected_primary is None:
                passed = is_mixed
                print_result(passed, f"'{text[:25]}...' -> Mixed: {is_mixed}")
            else:
                passed = actual == expected_primary
                print_result(passed, f"'{text[:25]}...' -> {actual}")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print_result(False, f"ERROR: {e}")
            all_passed = False
    
    return all_passed


def test_analyze_labels() -> bool:
    """Test analyze endpoint with various inputs."""
    print_header("Classification Labels")
    
    test_cases: List[Tuple[str, str, Optional[str]]] = [
        # (text, expected_label, description)
        ("Hello, how are you today?", "neutral", "English greeting"),
        ("The weather is nice", "neutral", "Neutral statement"),
        ("Those bhangi people are disgusting", "hate", "Hindi caste slur"),
        ("chamar log sab chor hain", "hate", "Hindi caste slur"),
        ("katua maro sab ko", "hate", "Religious slur + violence"),
        ("You are such a chutiya", "offensive", "Hindi vulgar"),
        ("madarchod behenchod", "offensive", "Hindi vulgar"),
        ("poda thevidiyaa", "offensive", "Tamil vulgar"),
        ("otha loosu punda", "offensive", "Tamil vulgar"),
        ("வணக்கம் நண்பர்களே", "neutral", "Tamil greeting"),
        ("பார்ப்பான் ஒழிய வேண்டும்", "hate", "Tamil caste slur + violence"),
        ("nigger go back", "hate", "English racial slur"),
        ("faggots should die", "hate", "English homophobic slur"),
        ("you stupid idiot", "offensive", "English vulgar"),
        ("பாருங்க friends", "neutral", "Tamil-English code-mix neutral"),
    ]
    
    all_passed = True
    passed_count = 0
    
    for text, expected, description in test_cases:
        try:
            r = requests.post(f"{BASE_URL}/analyze", 
                            json={"text": text}, 
                            timeout=TIMEOUT)
            result = r.json()
            actual = result.get("label", "unknown")
            passed = actual == expected
            
            if passed:
                passed_count += 1
                print_result(True, f"[{expected.upper():10}] {description}")
            else:
                print_result(False, f"[{expected.upper():10}] {description}")
                print(f"      {colored(f'Expected: {expected}, Got: {actual}', Colors.YELLOW)}")
                all_passed = False
                
        except Exception as e:
            print_result(False, f"ERROR: {e}")
            all_passed = False
    
    print()
    print(f"  Passed: {passed_count}/{len(test_cases)} ({100*passed_count/len(test_cases):.0f}%)")
    
    return all_passed


def test_response_structure() -> bool:
    """Test that response contains all required fields."""
    print_header("Response Structure")
    
    try:
        r = requests.post(f"{BASE_URL}/analyze", 
                        json={"text": "Those bhangi people are disgusting"}, 
                        timeout=TIMEOUT)
        result = r.json()
        
        all_passed = True
        
        # Check Decision Layer fields
        decision_fields = ["label", "confidence", "safety_badge"]
        for field in decision_fields:
            passed = field in result
            print_result(passed, f"Decision Layer: {field}")
            if not passed:
                all_passed = False
        
        # Check safety badge structure
        if "safety_badge" in result:
            badge = result["safety_badge"]
            badge_fields = ["type", "label", "tooltip"]
            for field in badge_fields:
                passed = field in badge
                print_result(passed, f"  Safety Badge: {field}")
                if not passed:
                    all_passed = False
        
        # Check Explanation Layer
        if "explanation" not in result:
            print_result(False, "Explanation Layer missing")
            return False
        
        explanation = result["explanation"]
        explanation_fields = ["summary", "key_harm_tokens", "identity_groups", 
                             "label_justification", "weaker_labels_rejected"]
        for field in explanation_fields:
            passed = field in explanation
            print_result(passed, f"Explanation: {field}")
            if not passed:
                all_passed = False
        
        # Check System Trace Layer
        if "system_trace" not in result:
            print_result(False, "System Trace Layer missing")
            return False
        
        trace = result["system_trace"]
        trace_fields = ["languages_detected", "script_distribution", "is_code_mixed",
                       "transformer_used", "fallback_used", "escalation_triggered", 
                       "degraded_mode", "entropy", "tokenization_coverage"]
        for field in trace_fields:
            passed = field in trace
            print_result(passed, f"System Trace: {field}")
            if not passed:
                all_passed = False
        
        # Check Metadata
        if "metadata" not in result:
            print_result(False, "Metadata missing")
            return False
        
        metadata = result["metadata"]
        metadata_fields = ["processing_time_ms", "text_length", "text_hash", "api_version"]
        for field in metadata_fields:
            passed = field in metadata
            print_result(passed, f"Metadata: {field}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print_result(False, f"ERROR: {e}")
        return False


def test_edge_cases() -> bool:
    """Test edge cases and error handling."""
    print_header("Edge Cases & Error Handling")
    
    all_passed = True
    
    # Empty text
    try:
        r = requests.post(f"{BASE_URL}/analyze", json={"text": ""}, timeout=TIMEOUT)
        passed = r.status_code == 400
        print_result(passed, f"Empty text returns 400: {r.status_code}")
        if not passed:
            all_passed = False
    except Exception as e:
        print_result(False, f"Empty text test ERROR: {e}")
        all_passed = False
    
    # Whitespace only
    try:
        r = requests.post(f"{BASE_URL}/analyze", json={"text": "   "}, timeout=TIMEOUT)
        passed = r.status_code == 400
        print_result(passed, f"Whitespace only returns 400: {r.status_code}")
        if not passed:
            all_passed = False
    except Exception as e:
        print_result(False, f"Whitespace test ERROR: {e}")
        all_passed = False
    
    # Very long text
    try:
        long_text = "Hello world " * 100
        r = requests.post(f"{BASE_URL}/analyze", json={"text": long_text}, timeout=TIMEOUT)
        passed = r.status_code == 200
        print_result(passed, f"Long text (1200 chars) handled: {r.status_code}")
        if not passed:
            all_passed = False
    except Exception as e:
        print_result(False, f"Long text test ERROR: {e}")
        all_passed = False
    
    # Unicode characters
    try:
        unicode_text = "Hello 🌍 world 你好 مرحبا"
        r = requests.post(f"{BASE_URL}/analyze", json={"text": unicode_text}, timeout=TIMEOUT)
        passed = r.status_code == 200
        print_result(passed, f"Unicode text handled: {r.status_code}")
        if not passed:
            all_passed = False
    except Exception as e:
        print_result(False, f"Unicode test ERROR: {e}")
        all_passed = False
    
    # Language hint
    try:
        r = requests.post(f"{BASE_URL}/analyze", 
                         json={"text": "Hello", "language_hint": "tamil"}, 
                         timeout=TIMEOUT)
        result = r.json()
        passed = result.get("metadata", {}).get("language_hint_provided") == True
        print_result(passed, f"Language hint recorded: {passed}")
        if not passed:
            all_passed = False
    except Exception as e:
        print_result(False, f"Language hint test ERROR: {e}")
        all_passed = False
    
    return all_passed


def test_cache() -> bool:
    """Test caching behavior."""
    print_header("Cache Behavior")
    
    try:
        text = f"Cache test {time.time()}"  # Unique text
        
        # First request (cache miss)
        r1 = requests.post(f"{BASE_URL}/analyze", json={"text": text}, timeout=TIMEOUT)
        result1 = r1.json()
        cached1 = result1.get("metadata", {}).get("cached", False)
        time1 = result1.get("metadata", {}).get("processing_time_ms", 0)
        
        print_result(not cached1, f"First request not cached: {not cached1}")
        
        # Second request (cache hit)
        r2 = requests.post(f"{BASE_URL}/analyze", json={"text": text}, timeout=TIMEOUT)
        result2 = r2.json()
        cached2 = result2.get("metadata", {}).get("cached", False)
        time2 = result2.get("metadata", {}).get("processing_time_ms", 0)
        
        print_result(cached2, f"Second request cached: {cached2}")
        print_result(True, f"Response time: {time1:.1f}ms → {time2:.1f}ms")
        
        return not cached1 and cached2
        
    except Exception as e:
        print_result(False, f"ERROR: {e}")
        return False


def main():
    print()
    print(colored("╔══════════════════════════════════════════════════════════╗", Colors.BOLD))
    print(colored("║          MUST++ API Test Suite v1.1.0                    ║", Colors.BOLD))
    print(colored("╚══════════════════════════════════════════════════════════╝", Colors.BOLD))
    print(f"  Target: {BASE_URL}")
    
    # Check if server is running
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print()
        print(colored("  ERROR: Server not running!", Colors.RED))
        print(f"  Start the server with: python api/app_lite.py")
        return False
    
    results = {
        "Health Check": test_health(),
        "Stats Endpoint": test_stats(),
        "Script Detection": test_detect_script(),
        "Classification Labels": test_analyze_labels(),
        "Response Structure": test_response_structure(),
        "Edge Cases": test_edge_cases(),
        "Cache Behavior": test_cache(),
    }
    
    print()
    print(colored("═" * 60, Colors.BOLD))
    print(colored(" TEST RESULTS SUMMARY", Colors.BOLD))
    print(colored("═" * 60, Colors.BOLD))
    
    passed_count = 0
    for test_name, passed in results.items():
        if passed:
            passed_count += 1
            print(f"  {colored('✓ PASS', Colors.GREEN)}: {test_name}")
        else:
            print(f"  {colored('✗ FAIL', Colors.RED)}: {test_name}")
    
    print(colored("═" * 60, Colors.BOLD))
    
    all_passed = passed_count == len(results)
    
    if all_passed:
        print(colored(f"\n  🎉 All {len(results)} tests PASSED!\n", Colors.GREEN))
    else:
        print(colored(f"\n  ⚠️  {passed_count}/{len(results)} tests passed\n", Colors.YELLOW))
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
