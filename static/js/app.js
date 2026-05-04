import { startAudioPlayerWorklet } from "./audio-player.js";
import { startAudioRecorderWorklet } from "./audio-recorder.js";

// Connection mapping required by ADK Backend
const userId = "demo-user";
const sessionId = "demo-session-" + Math.random().toString(36).substring(7);
let websocket = null;
let isConnected = false;
let isConnecting = false;

// Hardware Nodes
let audioPlayerNode;
let audioPlayerContext;
let audioRecorderNode;
let audioRecorderContext;
let micStream;

// Premium UI DOM Elements
const orbContainer = document.getElementById('orb-state');
const micText = document.getElementById('mic-text');
const socketStatus = document.getElementById('socket-status');
const orbButton = document.getElementById('orb-button');
const transcriptUser = document.getElementById('transcript-user');
const transcriptAi = document.getElementById('transcript-ai');
// End Button Elements
const endBtnContainer = document.getElementById('end-btn-container');
const endBtn = document.getElementById('end-btn');

const chatContainer = document.getElementById('chat-container');

const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');

// --- UI HELPERS ---

/**
 * Updates the glowing orb animation and text status.
 */
function setAgentState(state) {
    if (!orbContainer || !micText) return;
    
    // Reset classes
    orbContainer.className = 'orb-container';
    orbContainer.classList.add(`state-${state.toLowerCase()}`);
    
    const textMap = { 
        'IDLE': 'Tap to Connect', 
        'LISTENING': 'Listening...',
        'THINKING': 'Thinking...',
        'SPEAKING': 'Speaking...'
    };
    micText.innerText = textMap[state];
}

/**
 * Updates the connection status badge and resets the UI for a new session.
 */
function updateConnectionStatus(connected) {
    // Re-query these elements here to ensure they aren't null
    const statusBadge = document.getElementById('socket-status');
    const endContainer = document.getElementById('end-btn-container');

    if (connected) {
        if (statusBadge) {
            statusBadge.innerText = 'Online';
            statusBadge.classList.add('active');
        }
        setAgentState('LISTENING');
        
        // FORCING VISIBILITY HERE
        if (endContainer) {
            endContainer.style.opacity = "1";
            endContainer.style.pointerEvents = "auto";
            endContainer.classList.add('visible');
        }

        // Show Chat
        if (chatContainer) chatContainer.classList.add('visible');

        transcriptUser.innerText = "";
        transcriptAi.innerText = "";
    } else {
        if (statusBadge) {
            statusBadge.innerText = 'Offline';
            statusBadge.classList.remove('active');
        }
        setAgentState('IDLE');
        
        // HIDING HERE
        if (endContainer) {
            endContainer.style.opacity = "0";
            endContainer.style.pointerEvents = "none";
            endContainer.classList.remove('visible');
        }

        // Hide Chat
        if (chatContainer) chatContainer.classList.remove('visible');
        
        transcriptUser.innerText = "Press connect, then speak...";
        transcriptAi.innerText = "";
    }
}

/**
 * Decodes Base64 audio strings from the ADK payload into raw binary 
 * buffers that the AudioWorklet processor requires.
 */
function base64ToArray(base64) {
    let standardBase64 = base64.replace(/-/g, '+').replace(/_/g, '/');
    while (standardBase64.length % 4) standardBase64 += '=';
    
    const binaryString = window.atob(standardBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

/**
 * Cleans up spacing anomalies that can occur when the model 
 * streams CJK (Chinese, Japanese, Korean) characters.
 */
function cleanCJKSpaces(text) {
    const cjkPattern = /[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]/;
    return text.replace(/(\S)\s+(?=\S)/g, (match, char1) => {
        const nextCharMatch = text.match(new RegExp(char1 + '\\s+(.)', 'g'));
        if (nextCharMatch && nextCharMatch.length > 0) {
            const char2 = nextCharMatch[0].slice(-1);
            if (cjkPattern.test(char1) && cjkPattern.test(char2)) return char1;
        }
        return match;
    });
}

// --- CORE WEBSOCKET ROUTING ---

/**
 * Callback function passed to the Audio Recorder Worklet. 
 * Fires continuously to send raw mic chunks to the FastAPI backend.
 */
function audioRecorderHandler(pcmData) {
    if (websocket && websocket.readyState === WebSocket.OPEN && isConnected) {
        websocket.send(pcmData);
    }
}

/**
 * Primary connection function bound to the UI's Connect Button.
 */
async function connectWebsocket() {
    if (isConnected || isConnecting) return; // Prevent double-taps

    // 2. Set the lock
    isConnecting = true;
    updateConnectionStatus(false);

    // 1. Initialize ADK Hardware AudioWorklets (Handles Mic & Speakers)
    try {
        const [pNode, pCtx] = await startAudioPlayerWorklet();
        audioPlayerNode = pNode;
        audioPlayerContext = pCtx;

        // Listner to update Orb state after full audio playback from HW
        audioPlayerNode.port.onmessage = (event) => {
            console.log("4. MAIN THREAD: Caught flare from Worklet!", event.data);
            if (event.data.command === 'playbackComplete') {
                setAgentState('LISTENING');
            }
        };      
        // PROACTIVE FIX: Force the port to open (fixes Safari/WebKit bugs)
        // audioPlayerNode.port.start();  

        const [rNode, rCtx, stream] = await startAudioRecorderWorklet(audioRecorderHandler);
        audioRecorderNode = rNode;
        audioRecorderContext = rCtx;
        micStream = stream;
    } catch (err) {
        // --- Path C: Hardware/Permission Error ---
        alert("Connection failed: " + err.message);
        console.error(err);
        
        // CRITICAL: Clean up any half-started hardware
        if (micStream) micStream.getTracks().forEach(t => t.stop());
        
        isConnecting = false; // Reset lock so user can try again
    }

    // 2. Connect to ADK FastAPI WebSocket Route

    // Get settings from the UI
    const voice = document.getElementById('voice-select').value;
    const affectiveDialog = document.getElementById('affective-toggle').classList.contains('active');
    const proactiveAudio = document.getElementById('proactive-toggle').classList.contains('active');

    // Build query parameters safely
    const params = new URLSearchParams({
        voice: voice,
        affective_dialog: affectiveDialog,
        proactive_audio: proactiveAudio
    });

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${window.location.host}/live-smarthome-agent/ws/${userId}/${sessionId}?${params.toString()}`;
    websocket = new WebSocket(wsUrl);

    // Force binary messages to be ArrayBuffers, not Blobs, to receive audio stream directly. Does not affect Text Frames, only Binary Frames at the network protocol level.
    websocket.binaryType = "arraybuffer";    

    websocket.onopen = function () {
        isConnected = true;
        isConnecting = false;
        updateConnectionStatus(true);
    };

    let aiTurnActive = false;  

    websocket.onmessage = function (event) {
        // --- 1. INTERCEPT RAW BINARY AUDIO (Synchronous!) ---
        if (event.data instanceof ArrayBuffer) {
            if (audioPlayerNode) {
                // Send raw bytes straight to the worklet!
                audioPlayerNode.port.postMessage(event.data);
                setAgentState('SPEAKING');
            }
            return; // Stop processing this event, it's just audio.
        }

        // --- 2. EXISTING: PARSE JSON FOR EVERYTHING ELSE ---
        let adkEvent;
        try {
            adkEvent = JSON.parse(event.data);
        } catch (error) {
            console.warn("Received non-JSON payload, ignoring:", event.data);
            return; // Fail gracefully
        }

        // // 1. Handle custom wrapper messages from backend
        // if (adkEvent.type === "transcript") {
        //     if (adkEvent.role === "user") {
        //         transcriptUser.innerText = `"${adkEvent.text}"`;
        //     } else if (adkEvent.role === "ai") {
        //         transcriptAi.innerText = adkEvent.text;
        //     }
        //     return; // Safe to return here because custom messages don't contain audio or ADK state flags
        // }

        // ==========================================
        // EXTRACT DATA FIRST
        // ==========================================

        // -- User transcription --
        if (adkEvent.inputTranscription && adkEvent.inputTranscription.text) {
            let text = cleanCJKSpaces(adkEvent.inputTranscription.text);
            if (adkEvent.inputTranscription.finished) {                
                transcriptUser.innerText = `"${text}"`;
            } else {
                setAgentState('THINKING');
                transcriptUser.innerText = `"${text}..."`;
            }
        }

        // -- AI transcription --
        if (adkEvent.outputTranscription && adkEvent.outputTranscription.text) {
            const aiText = cleanCJKSpaces(adkEvent.outputTranscription.text);
            
            // Check if this is the final event (contains full string)
            if (adkEvent.outputTranscription.finished || !adkEvent.partial) {
                transcriptAi.innerText = aiText; // Overwrite to prevent duplication
            } else if (!aiTurnActive) {
                transcriptAi.innerText = aiText;
                aiTurnActive = true;
            } else {
                transcriptAi.innerText += aiText;
            }
        }        

        // 1. Safely check if the event contains 'content' and 'parts'
        if (adkEvent.content?.parts) {
            
            // 2. Loop through the parts (Gemini can sometimes return multiple tool calls in one event)
            adkEvent.content.parts.forEach((part, index) => {
                
                // 3A. Check if this part is a function call
                if (part.functionCall) {
                    // Define tools for logging
                    const logTools = [
                        "get_smart_home_devices_info",
                        "check_camera",
                        "control_airconditioner",
                        "control_camera",
                        "control_light",
                        "control_lock"                        
                    ];
                    // Log tool call requests
                    if (logTools.includes(part.functionCall.name)) {
                        window.logToolUse(part.functionCall.name, true, part.functionCall.args || {});
                    }
                                        
                }

                // 3B. Check if this part is a function response
                if (part.functionResponse) {
                    
                    // Define tools for logging
                    const logTools = [
                        "get_smart_home_devices_info",
                        "check_camera",
                        "control_airconditioner",
                        "control_camera",
                        "control_light",
                        "control_lock"                        
                    ];
                    // Log tool call requests
                    if (logTools.includes(part.functionResponse.name)) {
                        window.logToolUse(part.functionResponse.name, false, part.functionResponse.response || {});
                    }

                    // Define your active smart home tools for display update
                    const smartHomeTools = [
                        "control_airconditioner",
                        "control_camera",
                        "control_light",
                        "control_lock"
                    ];

                    // 4. If the response matches one of our device tools, trigger the UI
                    if (smartHomeTools.includes(part.functionResponse.name)) {
                        
                        // Extract the inner response payload
                        const deviceData = part.functionResponse.response;
                        
                        console.log(`✅ Device Event [${part.functionResponse.name}] Detected:`, deviceData);
                        
                        // 5. Fire the Dynamic Hero Stack animation!
                        // We check if it exists on the window object just to be safe
                        if (typeof window.addDynamicHeroAction === 'function') {
                            
                            // Base delay of 600ms (to allow audio catch up) + 300ms for each subsequent item
                            const baseDelay = 600;

                            // Stagger the animation using the loop index. 
                            setTimeout(() => {
                                window.addDynamicHeroAction(deviceData);
                            }, baseDelay + index * 300);
                            
                        } else {
                            console.warn("addDynamicHeroAction is not defined on the window object.");
                        }
                    }
                    // Catch the check_camera array response
                    else if (part.functionResponse.name === "check_camera") {
                        const payload = part.functionResponse.response;
                        console.log(`📹 Camera Check Event Detected:`, payload);
                        
                        if (typeof window.showCameraModal === 'function') {
                            // Trigger the new carousel modal we just built

                            // Base delay of 600ms (to allow audio catch up) 
                            const baseDelay = 600;

                            setTimeout(() => {
                                window.showCameraModal(payload);
                            }, baseDelay);                            
                            
                        } else {
                            console.warn("showCameraModal is not defined on the window object.");
                        }
                    }
                }
            });
        }

        // -- Audio playback --
        if (adkEvent.content && adkEvent.content.parts) {
            const audioParts = adkEvent.content.parts.filter(
                p => p.inlineData && p.inlineData.mimeType.startsWith("audio/pcm")
            );
            if (audioParts.length > 0 && audioPlayerNode) {
                for (const part of audioParts) {
                    audioPlayerNode.port.postMessage(base64ToArray(part.inlineData.data));
                }
                setAgentState('SPEAKING');
            }
        }

        // ==========================================
        // MANAGE STATE LAST
        // ==========================================

        // -- Turn complete --
        if (adkEvent.turnComplete) {
            // TRACER 1: Did the backend tell us the AI finished?
            console.log("1. MAIN THREAD: turnComplete received. Sending endOfTurn to Worklet.");            
            aiTurnActive = false;  // next AI chunk will start fresh
            if (audioPlayerNode) {
                // Tell the worklet to flush its buffer and report back
                audioPlayerNode.port.postMessage({ command: 'endOfTurn' });
            } else {
                // Fallback just in case hardware is missing
                setAgentState('LISTENING');
            }            
        }

        // -- Interruption --
        if (adkEvent.interrupted) {
            if (audioPlayerNode) {
                audioPlayerNode.port.postMessage({ command: "endOfAudio" });
            }
            aiTurnActive = false; // Reset so the next AI response starts fresh
            setAgentState('LISTENING');
        }
    };

    websocket.onclose = function () {
        isConnected = false; // FIX: Allows the user to reconnect without refreshing!
        isConnecting = false;
        updateConnectionStatus(false);
        
        // Release hardware resources
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }

        // 🛑 CRITICAL: Release Web Audio API Contexts
        if (audioPlayerContext && audioPlayerContext.state !== 'closed') {
            audioPlayerContext.close();
        }

        if (audioRecorderContext && audioRecorderContext.state !== 'closed') {
            audioRecorderContext.close();
        }        
    };

}

// ==========================================
// Text Chat Logic
// ==========================================
function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    if (websocket && websocket.readyState === WebSocket.OPEN && isConnected) {
        // Send the JSON format your main.py expects
        const payload = JSON.stringify({
            type: "text",
            text: text
        });
        
        websocket.send(payload);

        // Update user text (Gemini doesn't echo text inputs, so we must do it here)
        transcriptUser.innerText = `"${text}"`;        

        // Clear the text box
        chatInput.value = "";
    }
}

// Send on Button Click
if (chatSendBtn) {
    chatSendBtn.addEventListener('click', sendChatMessage);
}

// Send on Enter Key (Allow Shift+Enter for new lines)
if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); 
            sendChatMessage();
        }
    });
}

// ==========================================
// Session Termination Logic
// ==========================================
function endSession() {
    if (websocket && isConnected) {
        websocket.close(); // This automatically triggers websocket.onclose!
    }
}

// Bind connection and termination to buttons
if (orbButton) {
    orbButton.addEventListener('click', connectWebsocket);
}
if (endBtn) {
    endBtn.addEventListener('click', endSession);
}    