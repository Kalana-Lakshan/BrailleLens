import sys
import pyttsx3
import speech_recognition as sr

def speak(text: str) -> None:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        print(f"[TTS] {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS] {text} (Audio failed: {e})")

def listen_spoken_answer(timeout: int = 5) -> str | None:
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("[STT] Listening for your answer (speak a single letter)...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                print(f"[STT] You said: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                print("[STT] Timeout: No speech detected.")
                return None
            except sr.UnknownValueError:
                print("[STT] Error: Could not understand audio.")
                return None
            except sr.RequestError as e:
                print(f"[STT] Error: Could not request results; {e}")
                return None
    except Exception as e:
        print(f"[STT] Microphone unavailable: {e}")
        # Fallback to keyboard input
        try:
            val = input("[Fallback] Please type the character: ")
            return val.lower().strip()
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            return None
