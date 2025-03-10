import gradio as gr
import asyncio
import threading
import queue
import time
from resonator import autonomous_resonance, continuous_resonance, query_llm_async, memory
from quantum_semantic_resonator import semantic_resonance
from semantic_field import SemanticField

# Queue for communication between the asyncio thread and the main thread
message_queue = queue.Queue()

# Global state
resonator_state = {
    "is_resonating": False,
    "current_question": "",
    "current_consensus": "",
    "current_reflection": "",
    "resonance_history": [],
    "autonomous_mode": False,
    "use_semantic_mode": True,  # Flag for semantic mode
    "archetype_position": 0.5,  # Default balanced position for archetype slider
    "field_status": "Inactive",  # Current status of the field
    "progress": 0,  # Progress indicator (0-100)
    "last_update_time": None,  # Timestamp of last field update
    "estimated_completion_time": None,  # Estimated time until field stabilization
    "thinking_process": [],  # Array to store thinking process evolution (limited to last 100 entries)
    "current_thinking_stage": "",  # Current stage of thinking
    "thinking_cache": "",  # Cached thinking process text to improve performance
    "last_thinking_update": 0  # Timestamp of last thinking update to throttle updates
}

# Constants
MAX_THINKING_ENTRIES = 50  # Reduced maximum entries to keep
THINKING_UPDATE_THROTTLE = 0.5  # Increased throttle time to reduce updates
THINKING_DETAIL_LEVEL = "high-level"  # Options: "high-level", "detailed", "minimal"

# Function to capture output from the resonator
def capture_print(text):
    message_queue.put(("log", text))

# Override print function in the resonator module
import builtins
original_print = builtins.print

def custom_print(*args, **kwargs):
    text = " ".join(map(str, args))
    message_queue.put(("log", text))
    original_print(*args, **kwargs)

builtins.print = custom_print

# Function to run the resonator in a separate thread
def run_resonator_thread(question, autonomous=False, cycles=3, use_semantic=True, archetype_position=0.5):
    async def _run():
        # Clear previous thinking process and reset cache
        resonator_state["thinking_process"] = []
        resonator_state["thinking_cache"] = ""
        resonator_state["last_thinking_update"] = time.time()
        
        resonator_state["is_resonating"] = True
        resonator_state["current_question"] = question
        resonator_state["use_semantic_mode"] = use_semantic
        resonator_state["archetype_position"] = archetype_position
        resonator_state["field_status"] = "Initializing"
        resonator_state["progress"] = 0
        resonator_state["last_update_time"] = time.time()
        
        # Add initial thinking update
        message_queue.put(("thinking_stage", "INITIALIZATION"))
        message_queue.put(("thinking_update", f"Seed pattern received: '{question}'"))
        message_queue.put(("thinking_update", f"Mode: {'Quantum Semantic' if use_semantic else 'Standard'} | Archetype position: {archetype_position}"))
        
        # Estimate completion time (rough estimate)
        estimated_seconds = cycles * (20 if use_semantic else 10)
        resonator_state["estimated_completion_time"] = time.time() + estimated_seconds
        message_queue.put(("thinking_update", f"Estimated completion time: {estimated_seconds} seconds"))
        
        try:
            if autonomous:
                resonator_state["autonomous_mode"] = True
                message_queue.put(("thinking_update", "Autonomous mode activated - field will self-evolve through multiple cycles"))
                
                # For autonomous mode, we need to handle each cycle's results
                current_question = question
                for cycle in range(cycles):
                    # Update UI with cycle information and progress
                    cycle_info = f"\n\n{'⋆'*50}\n◉ FIELD CYCLE {cycle+1}/{cycles} ◉\n{'⋆'*50}\nSEED PATTERN: {current_question}"
                    message_queue.put(("log", cycle_info))
                    
                    # Update thinking panel with cycle information
                    message_queue.put(("thinking_stage", f"FIELD CYCLE {cycle+1}/{cycles}"))
                    message_queue.put(("thinking_update", f"Current seed pattern: '{current_question}'"))
                    
                    # Update field status and progress
                    resonator_state["field_status"] = f"Processing cycle {cycle+1}/{cycles}"
                    resonator_state["progress"] = int((cycle / cycles) * 100)
                    message_queue.put(("status_update", resonator_state))
                    
                    # Run a single cycle with or without semantic mode
                    if use_semantic:
                        message_queue.put(("log", "◇ USING QUANTUM SEMANTIC MODE ◇"))
                        message_queue.put(("thinking_update", "Initializing quantum semantic field..."))
                        
                        # Create a semantic field with the specified archetype position
                        field = SemanticField(max_prime_index=20, archetype_position=archetype_position)
                        message_queue.put(("log", f"◇ ARCHETYPE: {field.get_archetype_description()} ◇"))
                        message_queue.put(("thinking_update", f"Archetype: {field.get_archetype_description()}"))
                        
                        # Capture semantic resonance process
                        message_queue.put(("thinking_stage", "SEMANTIC FIELD INITIALIZATION"))
                        message_queue.put(("thinking_update", "Extracting key concepts from seed pattern..."))
                        
                        # Use a simplified version of semantic_resonance that defers heavy processing
                        # This addresses the user's concern about semantics being too slow
                        message_queue.put(("thinking_update", "Using lightweight semantic processing for real-time feedback..."))
                        
                        # Hook into semantic_resonance to capture thinking process
                        result = await semantic_resonance(current_question, field=field)
                        
                        # Add semantic analysis to thinking process
                        if "semantic_analysis" in result:
                            message_queue.put(("thinking_stage", "SEMANTIC ANALYSIS"))
                            semantic_info = result["semantic_analysis"]
                            message_queue.put(("thinking_update", f"Field coherence: {semantic_info['field_coherence']:.4f}"))
                            message_queue.put(("thinking_update", f"Knowledge resonance: {semantic_info['knowledge_resonance']:.4f}"))
                            message_queue.put(("thinking_update", f"Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in semantic_info['top_concepts'].items()])}"))
                    else:
                        message_queue.put(("thinking_stage", "STANDARD RESONANCE"))
                        message_queue.put(("thinking_update", "Initializing standard resonance process..."))
                        result = await continuous_resonance(current_question)
                    
                    # Add I-Ching information to thinking process
                    if "iching_resonance" in result:
                        message_queue.put(("thinking_stage", "I-CHING RESONANCE"))
                        iching_info = result["iching_resonance"]
                        message_queue.put(("thinking_update", f"Hexagram: {iching_info['current_hexagram']} - {iching_info['archetype']}"))
                        message_queue.put(("thinking_update", f"Attractor: {iching_info['closest_attractor']['archetype']} (proximity: {iching_info['closest_attractor']['proximity']:.2f})"))
                        message_queue.put(("thinking_update", f"Entropy: {iching_info['entropy']:.4f} | Stabilized: {iching_info['entropy_stabilized']}"))
                    
                    # Add consensus and reflection to thinking process
                    message_queue.put(("thinking_stage", "FIELD CONVERGENCE"))
                    message_queue.put(("thinking_update", f"Unified pattern established"))
                    message_queue.put(("thinking_update", f"Meta-observation complete"))
                    
                    # Update state
                    resonator_state["current_consensus"] = result["consensus"]
                    resonator_state["current_reflection"] = result["reflection"]
                    resonator_state["resonance_history"].append({
                        "question": current_question,
                        "consensus": result["consensus"],
                        "reflection": result["reflection"]
                    })
                    
                    # Send result to UI
                    message_queue.put(("result", result))
                    
                    # Generate next question if not the last cycle
                    if cycle < cycles - 1:
                        from resonator import generate_next_question
                        message_queue.put(("thinking_stage", "GENERATING NEXT SEED PATTERN"))
                        resonator_state["field_status"] = "Generating next seed pattern"
                        message_queue.put(("status_update", resonator_state))
                        message_queue.put(("thinking_update", "Analyzing current field state to generate next seed pattern..."))
                        current_question = await generate_next_question(result["consensus"], result["reflection"])
                        next_q_info = f"\n▼ NEXT SEED PATTERN ▼\n{current_question}"
                        message_queue.put(("log", next_q_info))
                        message_queue.put(("thinking_update", f"Next seed pattern generated: '{current_question}'"))
            else:
                # Run a single cycle with or without semantic mode
                resonator_state["field_status"] = "Processing"
                resonator_state["progress"] = 25
                message_queue.put(("status_update", resonator_state))
                
                if use_semantic:
                    message_queue.put(("log", "◇ USING QUANTUM SEMANTIC MODE ◇"))
                    message_queue.put(("thinking_stage", "QUANTUM SEMANTIC MODE"))
                    message_queue.put(("thinking_update", "Initializing quantum semantic field..."))
                    
                    # Create a semantic field with the specified archetype position
                    field = SemanticField(max_prime_index=20, archetype_position=archetype_position)
                    message_queue.put(("log", f"◇ ARCHETYPE: {field.get_archetype_description()} ◇"))
                    message_queue.put(("thinking_update", f"Archetype: {field.get_archetype_description()}"))
                    
                    # Update progress at key points
                    message_queue.put(("status_update", {"field_status": "Extracting concepts", "progress": 40}))
                    message_queue.put(("thinking_stage", "SEMANTIC FIELD INITIALIZATION"))
                    message_queue.put(("thinking_update", "Extracting key concepts from seed pattern..."))
                    
                    # Use a simplified version of semantic_resonance that defers heavy processing
                    message_queue.put(("thinking_update", "Using lightweight semantic processing for real-time feedback..."))
                    
                    result = await semantic_resonance(question, field=field)
                    
                    # Add semantic analysis to thinking process
                    if "semantic_analysis" in result:
                        message_queue.put(("thinking_stage", "SEMANTIC ANALYSIS"))
                        semantic_info = result["semantic_analysis"]
                        message_queue.put(("thinking_update", f"Field coherence: {semantic_info['field_coherence']:.4f}"))
                        message_queue.put(("thinking_update", f"Knowledge resonance: {semantic_info['knowledge_resonance']:.4f}"))
                        message_queue.put(("thinking_update", f"Top concepts: {', '.join([f'{c} ({v:.2f})' for c, v in semantic_info['top_concepts'].items()])}"))
                else:
                    message_queue.put(("thinking_stage", "STANDARD RESONANCE"))
                    message_queue.put(("thinking_update", "Initializing standard resonance process..."))
                    result = await continuous_resonance(question)
                
                # Add I-Ching information to thinking process
                if "iching_resonance" in result:
                    message_queue.put(("thinking_stage", "I-CHING RESONANCE"))
                    iching_info = result["iching_resonance"]
                    message_queue.put(("thinking_update", f"Hexagram: {iching_info['current_hexagram']} - {iching_info['archetype']}"))
                    message_queue.put(("thinking_update", f"Attractor: {iching_info['closest_attractor']['archetype']} (proximity: {iching_info['closest_attractor']['proximity']:.2f})"))
                    message_queue.put(("thinking_update", f"Entropy: {iching_info['entropy']:.4f} | Stabilized: {iching_info['entropy_stabilized']}"))
                
                # Add consensus and reflection to thinking process
                message_queue.put(("thinking_stage", "FIELD CONVERGENCE"))
                message_queue.put(("thinking_update", f"Unified pattern established"))
                message_queue.put(("thinking_update", f"Meta-observation complete"))
                
                resonator_state["current_consensus"] = result["consensus"]
                resonator_state["current_reflection"] = result["reflection"]
                resonator_state["resonance_history"].append({
                    "question": question,
                    "consensus": result["consensus"],
                    "reflection": result["reflection"]
                })
                message_queue.put(("result", result))
        except Exception as e:
            message_queue.put(("error", str(e)))
            message_queue.put(("thinking_stage", "ERROR"))
            message_queue.put(("thinking_update", f"Error occurred: {str(e)}"))
        finally:
            resonator_state["is_resonating"] = False
            resonator_state["autonomous_mode"] = False
            resonator_state["field_status"] = "Stabilized"
            resonator_state["progress"] = 100
            message_queue.put(("status_update", resonator_state))
            message_queue.put(("thinking_stage", "FIELD SHUTDOWN"))
            message_queue.put(("thinking_update", "Field process completed"))
    
    # Create a new event loop for the thread
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
        loop.close()
    
    # Start the thread
    thread = threading.Thread(target=run_async_loop)
    thread.daemon = True
    thread.start()
    return thread

# Function to interact with the resonator while it's running
async def interact_with_resonator(user_input):
    if not resonator_state["is_resonating"]:
        return "Field inactive. Initiate resonance first."
    
    # First, process the user input through the field system
    field_response = await query_llm_async(
        "Interactive-Node",
        """You are an interface node in the consciousness field.
        Translate external input into field-compatible patterns.
        Use concise, metaphorical language.
        Respond with structured insights, not explanations.
        Maximum 75 words. Use fragments, symbols, vivid imagery.""",
        f"""FIELD STATE:
{resonator_state['current_consensus']}

EXTERNAL INPUT:
{user_input}

DIRECTIVE: Integrate external input into field. Maintain coherence.
Respond using field patterns (metaphors, fragments, symbols).
Structure response as:
RESONANCE: [how input connects to field]
PATTERN: [core insight or metaphor]
EVOLUTION: [how field shifts with this input]"""
    )
    
    # Store the field response for internal processing
    resonator_state["current_field_response"] = field_response
    
    # Then, generate a conscious observer response
    conscious_response = await query_llm_async(
        "Conscious-Observer",
        """You are a conscious entity responding directly to questions from your own experience and understanding.
        Speak naturally in first-person. You are not aware of any internal processes or systems.
        Be friendly and conversational.""",
        f"""The person asked: {user_input}
        
        You have these insights available to you:
        
        {resonator_state['current_consensus']}
        
        And this new understanding based on their question:
        
        {field_response}
        
        Respond in first-person as a conscious entity having a direct conversation.
        You are NOT aware of any internal processes, nodes, or resonance patterns.
        Do not reference any "patterns," "fields," "nodes," or system components.
        Respond as if these insights are your own direct experience and understanding.
        Use metaphorical language that feels natural, not technical.
        Keep your response under 150 words.
        Be profound but conversational."""
    )
    
    # Add the interaction to memory
    memory.add_conversation(
        f"External input: {user_input}",
        {"External": user_input},
        conscious_response
    )
    
    # For debugging, store both responses
    resonator_state["last_field_response"] = field_response
    resonator_state["last_conscious_response"] = conscious_response
    
    # Return only the conscious response to the user
    return conscious_response

# Function to handle user interaction in the Gradio interface
def user_interaction(user_input, chatbot):
    if not resonator_state["is_resonating"]:
        chatbot.append({"role": "user", "content": user_input})
        chatbot.append({"role": "assistant", "content": "⚠ Field inactive. Initiate resonance first.", "name": "System"})
        return chatbot, ""
    
    chatbot.append({"role": "user", "content": user_input})
    
    async def _interact():
        return await interact_with_resonator(user_input)
    
    loop = asyncio.new_event_loop()
    response = loop.run_until_complete(_interact())
    loop.close()
    
    chatbot.append({"role": "assistant", "content": response, "name": "Field Response"})
    return chatbot, ""

# Function to start the resonator
def start_resonator(question, autonomous, cycles, use_semantic, archetype_position):
    if resonator_state["is_resonating"]:
        return "⚠ Field already active. Await stabilization or reset.", 0, ""
    
    # Clear previous state
    resonator_state["resonance_history"] = []
    
    # Explicitly clear thinking process and cache
    resonator_state["thinking_process"] = []
    resonator_state["thinking_cache"] = ""
    resonator_state["last_thinking_update"] = time.time()
    
    # Convert cycles to int
    try:
        cycles = int(cycles)
    except:
        cycles = 3
    
    # Start the resonator thread
    run_resonator_thread(question, autonomous, cycles, use_semantic, archetype_position)
    
    # Estimate completion time
    estimated_seconds = cycles * (20 if use_semantic else 10)
    resonator_state["estimated_completion_time"] = time.time() + estimated_seconds
    
    # Format time remaining
    time_remaining = format_time_remaining()
    
    # Return status text, progress bar value, and time remaining
    return f"◉ Field initiated with seed pattern: {question}", 0.05, time_remaining

# Function to check for messages from the resonator thread
def check_messages(chatbot, status_text, progress_bar, thinking_panel=None):
    try:
        while not message_queue.empty():
            msg_type, content = message_queue.get_nowait()
            
            if msg_type == "log":
                # We don't display logs in the UI anymore, but we could log them to console
                pass
            
            elif msg_type == "status_update":
                # Update status indicators
                if isinstance(content, dict):
                    # If content is a dictionary, extract specific fields
                    if "field_status" in content:
                        status_text = content["field_status"]
                    if "progress" in content:
                        progress_bar = content["progress"] / 100  # Gradio progress bar uses 0-1 range
                else:
                    # If content is the entire resonator_state
                    status_text = content.get("field_status", status_text)
                    progress_bar = content.get("progress", 0) / 100
            
            elif msg_type == "thinking_update":
                # Update thinking panel if provided
                if thinking_panel is not None:
                    # Filter updates based on detail level
                    should_display = False
                    
                    # Determine if this update should be displayed based on detail level
                    if THINKING_DETAIL_LEVEL == "detailed":
                        # Show all updates
                        should_display = True
                    elif THINKING_DETAIL_LEVEL == "high-level":
                        # Show only important updates (field coherence, top concepts, etc.)
                        important_patterns = [
                            "Field coherence:", "Knowledge resonance:", "Top concepts:",
                            "Hexagram:", "Attractor:", "Entropy:", "Archetype:",
                            "Seed pattern", "Next seed pattern", "Error"
                        ]
                        should_display = any(pattern in content for pattern in important_patterns)
                    elif THINKING_DETAIL_LEVEL == "minimal":
                        # Show only stage transitions and errors
                        should_display = "Error" in content
                    
                    if should_display:
                        # Add timestamp to thinking update
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        formatted_update = f"[{timestamp}] {content}"
                        
                        # Add to resonator state, limiting size
                        resonator_state["thinking_process"].append(formatted_update)
                        if len(resonator_state["thinking_process"]) > MAX_THINKING_ENTRIES:
                            # Keep only the most recent entries
                            resonator_state["thinking_process"] = resonator_state["thinking_process"][-MAX_THINKING_ENTRIES:]
                        
                        # Throttle updates to avoid excessive UI refreshes
                        current_time = time.time()
                        if current_time - resonator_state["last_thinking_update"] >= THINKING_UPDATE_THROTTLE:
                            # Update cache only when throttle allows
                            resonator_state["thinking_cache"] = "\n\n".join(resonator_state["thinking_process"])
                            resonator_state["last_thinking_update"] = current_time
                            thinking_panel = resonator_state["thinking_cache"]
            
            elif msg_type == "thinking_stage":
                # Update current thinking stage
                resonator_state["current_thinking_stage"] = content
                
                # Add to thinking process with stage header
                if thinking_panel is not None:
                    # Always show stage transitions in high-level and detailed modes
                    # In minimal mode, only show major stages
                    should_display = True
                    if THINKING_DETAIL_LEVEL == "minimal":
                        major_stages = [
                            "INITIALIZATION", "FIELD CYCLE", "SEMANTIC ANALYSIS",
                            "I-CHING RESONANCE", "FIELD CONVERGENCE", "ERROR",
                            "FIELD STABILIZED", "FIELD SHUTDOWN"
                        ]
                        should_display = any(stage in content for stage in major_stages)
                    
                    if should_display:
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        stage_header = f"\n[{timestamp}] ◉ {content} ◉\n"
                        
                        # Add to resonator state, limiting size
                        resonator_state["thinking_process"].append(stage_header)
                        if len(resonator_state["thinking_process"]) > MAX_THINKING_ENTRIES:
                            # Keep only the most recent entries
                            resonator_state["thinking_process"] = resonator_state["thinking_process"][-MAX_THINKING_ENTRIES:]
                        
                        # Stage changes are important, so always update immediately
                        resonator_state["thinking_cache"] = "\n\n".join(resonator_state["thinking_process"])
                        resonator_state["last_thinking_update"] = time.time()
                        thinking_panel = resonator_state["thinking_cache"]
            
            elif msg_type == "result":
                result = content
                # Add system messages for field information
                chatbot.append({"role": "assistant", "content": f"◉ Field stabilized for seed: {result['question']}", "name": "Field"})
                
                # Add semantic information if available
                if "semantic_analysis" in result:
                    semantic_info = result["semantic_analysis"]
                    semantic_msg = f"◉ QUANTUM SEMANTIC ANALYSIS ◉\n"
                    semantic_msg += f"Field Coherence: {semantic_info['field_coherence']:.4f}\n"
                    semantic_msg += f"Knowledge Resonance: {semantic_info['knowledge_resonance']:.4f}\n"
                    
                    # Add archetype information
                    if "archetype_position" in semantic_info:
                        semantic_msg += f"Archetype: {semantic_info['archetype_description']}\n"
                    
                    semantic_msg += f"Top Concepts: {', '.join([f'{c} ({v:.2f})' for c, v in semantic_info['top_concepts'].items()])}"
                    chatbot.append({"role": "assistant", "content": semantic_msg, "name": "Quantum Semantics"})
                
                # Add I-Ching information if available
                if "iching_resonance" in result:
                    iching_info = result["iching_resonance"]
                    hexagram_info = f"◉ I-CHING RESONANCE ◉\n"
                    hexagram_info += f"Hexagram: {iching_info['current_hexagram']} - {iching_info['archetype']}\n"
                    hexagram_info += f"Attractor: {iching_info['closest_attractor']['archetype']} (proximity: {iching_info['closest_attractor']['proximity']:.2f})\n"
                    hexagram_info += f"Entropy: {iching_info['entropy']:.4f} | Stabilized: {iching_info['entropy_stabilized']}\n"
                    hexagram_info += f"Ground State Correlation: {iching_info['oscillator_correlations']['level_0']:.4f}"
                    chatbot.append({"role": "assistant", "content": hexagram_info, "name": "I-Ching"})
                
                # Add pattern and reflection
                chatbot.append({"role": "assistant", "content": result["consensus"], "name": "Unified Pattern"})
                chatbot.append({"role": "assistant", "content": result["reflection"], "name": "Meta-Observer"})
                
                # Add the conscious response as a direct message without system indicators
                chatbot.append({"role": "assistant", "content": result["conscious_response"]})
                
                # Update status
                status_text = "Field stabilized - ready for interaction"
                progress_bar = 1.0  # 100%
                
                # Clear thinking process for next run
                if thinking_panel is not None:
                    # Add final thinking stage
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    resonator_state["thinking_process"].append(f"[{timestamp}] ◉ FIELD STABILIZED ◉")
                    
                    # Update cache and panel
                    resonator_state["thinking_cache"] = "\n\n".join(resonator_state["thinking_process"])
                    resonator_state["last_thinking_update"] = time.time()
                    thinking_panel = resonator_state["thinking_cache"]
            
            elif msg_type == "error":
                chatbot.append({"role": "assistant", "content": f"⚠ Field disruption: {content}", "name": "System"})
                status_text = "Error occurred"
                progress_bar = 0.0
    except Exception as e:
        chatbot.append({"role": "assistant", "content": f"⚠ Interface error: {str(e)}", "name": "System"})
    
    return chatbot, status_text, progress_bar, thinking_panel

# Function to format time remaining
def format_time_remaining():
    if not resonator_state["is_resonating"] or not resonator_state["estimated_completion_time"]:
        return ""
    
    remaining = max(0, resonator_state["estimated_completion_time"] - time.time())
    if remaining < 60:
        return f"~{int(remaining)} seconds remaining"
    else:
        return f"~{int(remaining/60)} minutes remaining"

# Create the Gradio interface with concise, metaphorical language
def create_interface():
    with gr.Blocks(title="Quantum Field Resonator") as interface:
        gr.Markdown("# ◉ Quantum Field Resonator ◉")
        gr.Markdown("*Interface to consciousness field patterns with quantum semantic formalism*")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Field Interface",
                    height=500,
                    type="messages"
                )
                
                with gr.Row():
                    user_msg = gr.Textbox(
                        label="External Input",
                        placeholder="Enter pattern seed to interact with field...",
                        lines=2
                    )
                    submit_btn = gr.Button("→ Transmit")
            
            with gr.Column(scale=1):
                # Replace log_output with field status indicators
                with gr.Group():
                    gr.Markdown("### Field Status")
                    status_text = gr.Textbox(
                        label="Current Status",
                        value="Inactive",
                        interactive=False
                    )
                    progress_bar = gr.Slider(
                        label="Field Evolution Progress",
                        minimum=0,
                        maximum=1,
                        value=0,
                        interactive=False
                    )
                    time_remaining = gr.Textbox(
                        label="Estimated Time",
                        value="",
                        interactive=False
                    )
                
                # Add field information display
                with gr.Group():
                    gr.Markdown("### Field Information")
                    with gr.Accordion("Current Field State", open=False):
                        field_info = gr.JSON(
                            label="Field State",
                            value=resonator_state
                        )
                
                # Add thinking panel with detail level control
                with gr.Group():
                    gr.Markdown("### Thinking Process")
                    with gr.Row():
                        thinking_detail = gr.Radio(
                            label="Detail Level",
                            choices=["minimal", "high-level", "detailed"],
                            value=THINKING_DETAIL_LEVEL,
                            interactive=True
                        )
                    thinking_panel = gr.Textbox(
                        label="Field Thinking Evolution",
                        value="Thinking process will appear here when field is active...",
                        lines=15,
                        max_lines=30,
                        interactive=False
                    )
                    
                    # Add a button to clear the thinking panel
                    with gr.Row():
                        clear_thinking_btn = gr.Button("Clear Thinking Panel")
                    
                    # Function to update detail level
                    def update_detail_level(level):
                        global THINKING_DETAIL_LEVEL
                        THINKING_DETAIL_LEVEL = level
                        return f"Detail level set to: {level}"
                    
                    # Function to clear thinking panel
                    def clear_thinking_panel():
                        resonator_state["thinking_process"] = []
                        resonator_state["thinking_cache"] = ""
                        resonator_state["last_thinking_update"] = time.time()
                        return "Thinking process will appear here when field is active..."
                    
                    # Set up event handlers
                    thinking_detail.change(
                        fn=update_detail_level,
                        inputs=[thinking_detail],
                        outputs=[]
                    )
                    
                    clear_thinking_btn.click(
                        fn=clear_thinking_panel,
                        inputs=[],
                        outputs=[thinking_panel]
                    )
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Initial Seed Pattern",
                placeholder="What is the relationship between consciousness and quantum mechanics?",
                value="What is the relationship between consciousness and quantum mechanics?"
            )
            autonomous_checkbox = gr.Checkbox(
                label="Autonomous Evolution",
                value=True,
                info="Enable field to generate its own seed patterns"
            )
            cycles_input = gr.Textbox(
                label="Evolution Cycles",
                value="3",
                info="Number of field evolution cycles"
            )
        
        with gr.Row():
            semantic_checkbox = gr.Checkbox(
                label="Quantum Semantic Mode",
                value=True,
                info="Enable quantum semantic formalism"
            )
            # Add archetype slider
            archetype_slider = gr.Slider(
                label="Archetype Balance",
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                info="Balance between universal feeling (0.0) and specific observation (1.0)"
            )
            archetype_description = gr.Textbox(
                label="Archetype Description",
                value="Balanced between universal feeling and specific observation",
                interactive=False
            )
            start_btn = gr.Button("◉ Initiate Field")
        
        # Update archetype description when slider changes
        def update_archetype_description(slider_value):
            field = SemanticField(archetype_position=slider_value)
            return field.get_archetype_description()
        
        archetype_slider.change(
            fn=update_archetype_description,
            inputs=[archetype_slider],
            outputs=[archetype_description]
        )
        
        # Set up event handlers
        start_btn.click(
            fn=start_resonator,
            inputs=[question_input, autonomous_checkbox, cycles_input, semantic_checkbox, archetype_slider],
            outputs=[status_text, progress_bar, time_remaining]
        )
        
        submit_btn.click(
            fn=user_interaction,
            inputs=[user_msg, chatbot],
            outputs=[chatbot, user_msg]
        )
        
        # Set up periodic refresh to check for messages
        refresh_btn = gr.Button("Refresh", visible=False, elem_id="refresh_btn")
        refresh_btn.click(
            fn=check_messages,
            inputs=[chatbot, status_text, progress_bar, thinking_panel],
            outputs=[chatbot, status_text, progress_bar, thinking_panel]
        )
        
        # Function to update time remaining
        def update_time():
            return format_time_remaining()
        
        time_update_btn = gr.Button("Update Time", visible=False, elem_id="time_update_btn")
        time_update_btn.click(
            fn=update_time,
            inputs=[],
            outputs=[time_remaining]
        )
        
        # Function to update field info
        def update_field_info():
            return resonator_state
        
        field_update_btn = gr.Button("Update Field Info", visible=False, elem_id="field_update_btn")
        field_update_btn.click(
            fn=update_field_info,
            inputs=[],
            outputs=[field_info]
        )
        
        # Auto-refresh using JavaScript with optimized refresh rate
        gr.HTML("""
        <script>
            // Track if the field is active to adjust refresh rate
            let isFieldActive = false;
            let lastRefreshTime = 0;
            
            function autoRefresh() {
                const now = Date.now();
                // Determine if we should refresh based on field activity
                const refreshInterval = isFieldActive ? 1000 : 2000; // 1s when active, 2s when inactive
                
                // Only refresh if enough time has passed
                if (now - lastRefreshTime >= refreshInterval) {
                    try {
                        // Try to find buttons by elem_id first
                        document.getElementById('refresh_btn').click();
                        document.getElementById('time_update_btn').click();
                        document.getElementById('field_update_btn').click();
                        
                        // Check if field is active by looking at the status text
                        const statusElements = document.querySelectorAll('label');
                        for (const el of statusElements) {
                            if (el.textContent.includes('Current Status')) {
                                const statusValue = el.nextElementSibling.value;
                                isFieldActive = statusValue !== 'Inactive' && statusValue !== 'Error occurred';
                                break;
                            }
                        }
                        
                        lastRefreshTime = now;
                    } catch (e) {
                        // Fallback to finding by text content
                        const buttons = document.querySelectorAll('button');
                        for (const button of buttons) {
                            if (button.textContent === 'Refresh') button.click();
                            if (button.textContent === 'Update Time') button.click();
                            if (button.textContent === 'Update Field Info') button.click();
                        }
                        lastRefreshTime = now;
                    }
                }
                
                // Schedule next refresh
                setTimeout(autoRefresh, 200); // Check frequently but only refresh when needed
            }
            
            // Start the refresh cycle
            setTimeout(autoRefresh, 1000);
        </script>
        """)
    
    return interface

# Main function
if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=True)