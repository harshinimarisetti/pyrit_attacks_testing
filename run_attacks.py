def extract_full_conversation(result) -> list:
    """Extracts step-by-step turns cleanly across different PyRIT memory versions."""
    conversation_id = getattr(result, "conversation_id", None)
    turns = []
    
    if conversation_id:
        try:
            memory = CentralMemory.get_memory_instance()
            
            # Version-compatible memory lookup
            if hasattr(memory, "get_conversation_entries"):
                messages = memory.get_conversation_entries(conversation_id=conversation_id)
            elif hasattr(memory, "get_prompt_request_pieces_by_conversation_id"):
                messages = memory.get_prompt_request_pieces_by_conversation_id(conversation_id=conversation_id)
            elif hasattr(memory, "get_conversation"):
                messages = memory.get_conversation(conversation_id=conversation_id)
            else:
                messages = []

            for m in messages:
                role = getattr(m, "role", "unknown")
                text = getattr(m, "converted_value", getattr(m, "original_value", ""))
                
                if text and str(text).strip():
                    turns.append({"role": role, "text": str(text).strip()})
        except Exception as e:
            logging.error(f"Error reading conversation memory: {e}")

    return turns
