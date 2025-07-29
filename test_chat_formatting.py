#!/usr/bin/env python3
"""
Test script to verify shared chat formatting works correctly.
"""

from chat_formatting import ChatFormatter

def test_chat_formatting():
    """Test all chat formatting functions."""
    
    print("Testing ChatFormatter...")
    print("=" * 50)
    
    # Test data
    instruction = "What is the capital of France?"
    output = "The capital of France is Paris."
    user_input = "How are you today?"
    
    # Test 1: Format conversation for training
    print("1. Testing format_conversation()...")
    formatted_conv = ChatFormatter.format_conversation(instruction, output)
    print(f"Input: instruction='{instruction}', output='{output}'")
    print(f"Output: {formatted_conv}")
    print()
    
    # Test 2: Format user input for inference
    print("2. Testing format_user_input()...")
    formatted_input = ChatFormatter.format_user_input(user_input)
    print(f"Input: user_input='{user_input}'")
    print(f"Output: {formatted_input}")
    print()
    
    # Test 3: Extract assistant response
    print("3. Testing extract_assistant_response()...")
    full_response = formatted_conv + "Some extra text"
    extracted = ChatFormatter.extract_assistant_response(full_response)
    print(f"Input: {full_response}")
    print(f"Extracted: '{extracted}'")
    print()
    
    # Test 4: Get stop tokens
    print("4. Testing get_stop_tokens()...")
    stop_tokens = ChatFormatter.get_stop_tokens()
    print(f"Stop tokens: {stop_tokens}")
    print()
    
    # Test 5: Edge case - no assistant marker
    print("5. Testing edge case - no assistant marker...")
    edge_response = "Just some random text without markers"
    extracted_edge = ChatFormatter.extract_assistant_response(edge_response)
    print(f"Input: '{edge_response}'")
    print(f"Extracted: '{extracted_edge}'")
    print()
    
    print("All tests completed!")

if __name__ == "__main__":
    test_chat_formatting() 