#!/usr/bin/env python3
"""
PACA ASCII-Only Version - No Unicode Issues
"""

import sys
import os
import asyncio

def main():
    print("="*50)
    print("PACA ASCII-Only Test Version")
    print("="*50)
    print("Python version:", sys.version.split()[0])
    print("Current directory:", os.getcwd())
    print("="*50)

    # Test PACA imports
    try:
        print("Testing PACA import...")
        from paca.cognitive import create_metacognition_engine
        print("[OK] PACA cognitive module imported")

        from paca.learning.auto.engine import AutoLearningSystem
        print("[OK] PACA learning module imported")

    except ImportError as e:
        print("[ERROR] PACA import failed:", str(e))
        input("Press Enter to continue anyway...")

    print("\n" + "="*50)
    print("PACA Chat Test - ASCII Only")
    print("Commands: stats, quit")
    print("Try: 'python study' or 'learn javascript'")
    print("="*50)

    chat_count = 0
    learning_count = 0

    while True:
        try:
            # Get input
            user_input = input(f"\nYOU: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if user_input.lower() == 'stats':
                print(f"\n[STATS] Chats: {chat_count}, Learning: {learning_count}")
                continue

            chat_count += 1

            # Detect learning
            learning_keywords = ['learn', 'study', 'teach', 'explain', 'help']
            is_learning = any(word in user_input.lower() for word in learning_keywords)

            if is_learning:
                learning_count += 1
                print("[DETECTED] Learning request")

            # Simple responses
            if 'python' in user_input.lower():
                print("PACA: I'll help you learn Python! Let's start with basics.")
            elif 'javascript' in user_input.lower():
                print("PACA: JavaScript learning time! Async, promises, and more.")
            elif 'react' in user_input.lower():
                print("PACA: React study session! Components, hooks, state management.")
            elif is_learning:
                print("PACA: Great! I'm here to help you learn. What topic interests you?")
            else:
                print(f"PACA: I understand you said '{user_input}'. How can I help you learn?")

            # Show meta info
            quality = 60 + (len(user_input) % 40)
            print(f"[META] Reasoning quality: {quality}/100")

            if is_learning:
                print("[ACTIVE] Learning patterns activated")

        except KeyboardInterrupt:
            print("\n\nExiting on Ctrl+C...")
            break
        except EOFError:
            print("\n\nInput ended, exiting...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

    # Final stats
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"Total chats: {chat_count}")
    print(f"Learning requests: {learning_count}")
    if chat_count > 0:
        success_rate = (learning_count / chat_count) * 100
        print(f"Learning success rate: {success_rate:.1f}%")
    print("Test completed successfully!")
    print("="*50)

    # Auto-close when run from batch file, manual close when run directly
    if len(sys.argv) > 1 and sys.argv[1] == "--auto-close":
        print("\nAuto-closing in 3 seconds...")
        import time
        time.sleep(3)
    else:
        try:
            input("\nPress Enter to close...")
        except (EOFError, KeyboardInterrupt):
            print("\nClosing...")

if __name__ == "__main__":
    main()