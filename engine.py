# File: engine.py
# Core TTS model loading and speech generation logic.

import logging
import random
import numpy as np
import torch
import os
import time
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    Uses safe error handling to avoid cascading CUDA failures.
    """
    try:
        torch.manual_seed(seed_value)
        
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
            except RuntimeError as e:
                error_str = str(e).lower()
                if "device-side assert" in error_str or "assertion" in error_str:
                    logger.warning(f"Cannot set CUDA seed due to device-side assertion: {e}")
                else:
                    logger.warning(f"CUDA seed setting failed: {e}")
        
        if torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed_value)
            except Exception as e:
                logger.warning(f"MPS seed setting failed: {e}")
        
        random.seed(seed_value)
        np.random.seed(seed_value)
        logger.info(f"Global seed set to: {seed_value}")
        
    except Exception as e:
        logger.error(f"Failed to set global seed: {e}")
        # Don't raise exception, just log the error and continue


def _setup_cuda_debugging():
    """
    Sets up CUDA debugging environment if enabled in configuration.
    """
    if config_manager.get_bool("debug.cuda_launch_blocking", False):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info("CUDA_LAUNCH_BLOCKING enabled for better error tracking")
    
    if config_manager.get_bool("debug.verbose_error_logging", True):
        # Enable additional CUDA error context
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some memory margin
                logger.info("CUDA memory fraction set to 0.95 to prevent OOM errors")
            except Exception as e:
                logger.warning(f"Could not set CUDA memory fraction: {e}")


def _clear_cuda_cache():
    """
    Clears CUDA cache to recover from potential memory issues.
    Uses safe error handling to avoid cascading CUDA failures.
    """
    if not torch.cuda.is_available():
        return
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("CUDA cache cleared successfully")
    except RuntimeError as e:
        error_str = str(e).lower()
        if "device-side assert" in error_str or "assertion" in error_str:
            logger.warning("Cannot clear CUDA cache due to device-side assertion - CUDA device in error state")
        else:
            logger.warning(f"CUDA cache clearing failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error clearing CUDA cache: {e}")


def _validate_generation_inputs(text: str, audio_prompt_path: Optional[str]) -> bool:
    """
    Validates inputs before passing them to the model generation.
    
    Args:
        text: The text to synthesize
        audio_prompt_path: Path to audio prompt file
        
    Returns:
        bool: True if inputs are valid, False otherwise
    """
    if not text or not text.strip():
        logger.error("Text input is empty or contains only whitespace")
        return False
    
    # Clean and normalize the text
    text_clean = text.strip()
    
    # Check for reasonable text length
    if len(text_clean) > 10000:  # Reasonable limit
        logger.warning(f"Text input is very long ({len(text_clean)} characters), this may cause issues")
    
    # Check for minimum meaningful text length
    if len(text_clean) < 1:
        logger.error("Text input is too short to generate meaningful audio")
        return False
    
    # Validate text contains some alphanumeric characters
    import re
    if not re.search(r'[a-zA-Z0-9]', text_clean):
        logger.error("Text input contains no alphanumeric characters")
        return False
    
    # Validate audio prompt path if provided
    if audio_prompt_path:
        if not os.path.exists(audio_prompt_path):
            logger.error(f"Audio prompt path does not exist: {audio_prompt_path}")
            return False
        
        # Check file size is reasonable
        try:
            file_size = os.path.getsize(audio_prompt_path)
            if file_size == 0:
                logger.error(f"Audio prompt file is empty: {audio_prompt_path}")
                return False
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"Audio prompt file is very large ({file_size / 1024 / 1024:.1f}MB): {audio_prompt_path}")
        except OSError as e:
            logger.error(f"Cannot access audio prompt file: {audio_prompt_path}, error: {e}")
            return False
    
    return True


def _validate_cuda_state() -> bool:
    """
    Validates that CUDA is in a working state before attempting generation.
    
    Returns:
        bool: True if CUDA is working or not being used, False if CUDA is in error state
    """
    if not torch.cuda.is_available():
        return True  # CPU mode is fine
    
    try:
        # Test basic CUDA operations
        test_tensor = torch.tensor([1.0], device='cuda')
        test_result = test_tensor + 1.0
        test_result.cpu()  # Move back to CPU
        return True
    except RuntimeError as e:
        error_str = str(e).lower()
        if "device-side assert" in error_str or "assertion" in error_str:
            logger.error("CUDA device is in assertion error state - cannot proceed with CUDA operations")
            return False
        else:
            logger.warning(f"CUDA functionality test failed: {e}")
            return False
    except Exception as e:
        logger.warning(f"Unexpected error testing CUDA state: {e}")
        return False


def _handle_cuda_error(error: Exception, retry_count: int) -> bool:
    """
    Handles CUDA-specific errors and determines if retry is appropriate.
    
    Args:
        error: The exception that occurred
        retry_count: Current retry attempt number
        
    Returns:
        bool: True if retry should be attempted, False otherwise
    """
    error_str = str(error).lower()
    
    # Check for known CUDA assertion errors - these are critical and require special handling
    if "device-side assert" in error_str or "assertion" in error_str:
        logger.error(f"CUDA device-side assertion detected: {error}")
        
        if config_manager.get_bool("debug.enable_cuda_error_recovery", True):
            logger.info("Attempting CUDA error recovery for device-side assertion...")
            
            # For device-side assertions, we cannot safely use most CUDA operations
            # Try safe cache clearing first
            _clear_cuda_cache()
            
            # Skip CUDA synchronization if it's likely to fail due to assertion state
            # Device-side assertions often leave CUDA in an unrecoverable state
            logger.warning("Device-side assertion detected - CUDA device may be in unstable state")
            
            # For device-side assertions, be more conservative with retries
            max_retries = max(1, config_manager.get_int("debug.max_generation_retries", 2) // 2)
            return retry_count < max_retries
    
    # Check for CUDA out of memory errors
    elif "out of memory" in error_str or "cuda oom" in error_str:
        logger.error(f"CUDA out of memory error: {error}")
        _clear_cuda_cache()
        return retry_count < config_manager.get_int("debug.max_generation_retries", 2)
    
    # Check for other CUDA runtime errors
    elif "cuda" in error_str or "gpu" in error_str:
        logger.error(f"CUDA runtime error: {error}")
        _clear_cuda_cache()
        
        # Try CUDA synchronization for non-assertion errors
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                logger.info("CUDA synchronization completed")
            except Exception as sync_error:
                logger.warning(f"CUDA synchronization failed: {sync_error}")
        
        return retry_count < config_manager.get_int("debug.max_generation_retries", 2)
    
    return False


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Setup CUDA debugging if enabled
        _setup_cuda_debugging()
        
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")

        # Get configured model_repo_id for logging and context,
        # though from_pretrained might use its own internal default if not overridden.
        model_repo_id_config = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )

        logger.info(
            f"Attempting to load model directly using from_pretrained (expected from Hugging Face repository: {model_repo_id_config} or library default)."
        )
        try:
            # Directly use from_pretrained. This will utilize the standard Hugging Face cache.
            # The ChatterboxTTS.from_pretrained method handles downloading if the model is not in the cache.
            chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
            # The actual repo ID used by from_pretrained is often internal to the library,
            # but logging the configured one provides user context.
            logger.info(
                f"Successfully loaded TTS model using from_pretrained on {model_device} (expected from '{model_repo_id_config}' or library default)."
            )
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained (expected from '{model_repo_id_config}' or library default): {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model with robust error handling.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model, model_device, MODEL_LOADED

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    # Validate inputs before proceeding
    if not _validate_generation_inputs(text, audio_prompt_path):
        logger.error("Input validation failed")
        return None, None

    max_retries = config_manager.get_int("debug.max_generation_retries", 2)
    retry_count = 0
    original_device = model_device
    
    while retry_count <= max_retries:
        try:
            # Pre-generation CUDA state validation for device-side assertion detection
            if model_device == "cuda" and torch.cuda.is_available():
                if not _validate_cuda_state():
                    logger.error("CUDA device is in error state, cannot proceed with CUDA generation")
                    # If we're on CUDA and it's in error state, trigger CPU fallback
                    raise RuntimeError("CUDA device-side assertion state detected")
                
                try:
                    # Check CUDA state before generation
                    torch.cuda.synchronize()
                    memory_allocated = torch.cuda.memory_allocated()
                    memory_reserved = torch.cuda.memory_reserved()
                    logger.debug(f"CUDA memory before generation: allocated={memory_allocated/1024**2:.1f}MB, reserved={memory_reserved/1024**2:.1f}MB")
                except Exception as cuda_check_error:
                    error_str = str(cuda_check_error).lower()
                    if "device-side assert" in error_str or "assertion" in error_str:
                        logger.error("CUDA device-side assertion detected during pre-check")
                        raise RuntimeError("CUDA device-side assertion state detected")
                    else:
                        logger.warning(f"CUDA state check failed: {cuda_check_error}")

            # Set seed globally if a specific seed value is provided and is non-zero.
            if seed != 0:
                logger.info(f"Applying user-provided seed for generation: {seed}")
                set_seed(seed)
            else:
                logger.info(
                    "Using default (potentially random) generation behavior as seed is 0."
                )

            logger.debug(
                f"Synthesizing with params (attempt {retry_count + 1}): audio_prompt='{audio_prompt_path}', temp={temperature}, "
                f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, device={model_device}"
            )

            # Call the core model's generate method with timeout protection
            start_time = time.time()
            
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed successfully in {generation_time:.2f} seconds")

            # Validate output tensor
            if wav_tensor is None:
                raise RuntimeError("Model generated None output")
            
            if not isinstance(wav_tensor, torch.Tensor):
                raise RuntimeError(f"Model output is not a tensor: {type(wav_tensor)}")
            
            if wav_tensor.numel() == 0:
                raise RuntimeError("Model generated empty tensor")
            
            # Check for NaN or infinite values
            if torch.isnan(wav_tensor).any():
                raise RuntimeError("Model generated tensor contains NaN values")
            
            if torch.isinf(wav_tensor).any():
                raise RuntimeError("Model generated tensor contains infinite values")

            # The ChatterboxTTS.generate method already returns a CPU tensor.
            logger.info(f"Synthesis successful - tensor shape: {wav_tensor.shape}, device: {wav_tensor.device}")
            return wav_tensor, chatterbox_model.sr

        except Exception as e:
            error_message = str(e)
            retry_count += 1
            
            if config_manager.get_bool("debug.verbose_error_logging", True):
                logger.error(f"Error during TTS synthesis (attempt {retry_count}/{max_retries + 1}): {e}", exc_info=True)
            else:
                logger.error(f"Error during TTS synthesis (attempt {retry_count}/{max_retries + 1}): {e}")

            # Handle CUDA-specific errors with enhanced device-side assertion handling
            if "cuda" in error_message.lower() or "gpu" in error_message.lower():
                # Detect device-side assertion errors specifically 
                is_device_assertion = ("device-side assert" in error_message.lower() or 
                                     "assertion" in error_message.lower())
                
                if is_device_assertion:
                    logger.error("Device-side assertion error detected - CUDA device is in critical error state")
                    
                    # For device-side assertions, immediately attempt CPU fallback if enabled
                    if (config_manager.get_bool("debug.fallback_to_cpu_on_cuda_error", True) and 
                        original_device != "cpu"):
                        logger.warning("Device-side assertion detected - attempting immediate CPU fallback")
                        
                        try:
                            # Reload model on CPU without attempting CUDA operations
                            old_device_setting = config_manager.get_string("tts_engine.device", "auto")
                            config_manager.data["tts_engine"]["device"] = "cpu"
                            
                            # Safe CUDA cache clearing (will handle assertion errors gracefully)
                            _clear_cuda_cache()
                            
                            MODEL_LOADED = False
                            chatterbox_model = None
                            
                            if load_model():
                                logger.info("Successfully reloaded model on CPU after device-side assertion")
                                model_device = "cpu"
                                # Reset retry count for CPU attempt
                                retry_count = 0
                                continue
                            else:
                                logger.error("Failed to reload model on CPU after device-side assertion")
                                # Restore original device setting
                                config_manager.data["tts_engine"]["device"] = old_device_setting
                                
                        except Exception as fallback_error:
                            logger.error(f"CPU fallback failed after device-side assertion: {fallback_error}")
                            # Restore original device setting
                            config_manager.data["tts_engine"]["device"] = old_device_setting
                        
                        # If CPU fallback failed, don't retry with CUDA in assertion state
                        logger.error("Cannot recover from device-side assertion - stopping generation attempts")
                        break
                else:
                    # Handle other CUDA errors with normal retry logic
                    if _handle_cuda_error(e, retry_count - 1):
                        logger.info(f"Retrying generation after CUDA error recovery (attempt {retry_count + 1})")
                        
                        # Add a small delay to allow GPU to stabilize
                        time.sleep(0.5)
                        continue
                    else:
                        # Check if we should fallback to CPU for persistent non-assertion errors
                        if (config_manager.get_bool("debug.fallback_to_cpu_on_cuda_error", True) and 
                            original_device != "cpu" and retry_count > max_retries // 2):
                            
                            logger.warning("Attempting CPU fallback after persistent CUDA errors")
                            try:
                                # Reload model on CPU
                                old_device_setting = config_manager.get_string("tts_engine.device", "auto")
                                config_manager.data["tts_engine"]["device"] = "cpu"
                                
                                # Clear CUDA cache and reload
                                _clear_cuda_cache()
                                
                                MODEL_LOADED = False
                                chatterbox_model = None
                                
                                if load_model():
                                    logger.info("Successfully reloaded model on CPU for fallback")
                                    model_device = "cpu"
                                    continue
                                else:
                                    logger.error("Failed to reload model on CPU")
                                    # Restore original device setting
                                    config_manager.data["tts_engine"]["device"] = old_device_setting
                                    
                            except Exception as fallback_error:
                                logger.error(f"CPU fallback failed: {fallback_error}")
                                # Restore original device setting
                                config_manager.data["tts_engine"]["device"] = old_device_setting
            
            # For non-CUDA errors or if retries exhausted
            if retry_count > max_retries:
                logger.error(f"Synthesis failed after {max_retries + 1} attempts")
                break
            else:
                logger.info(f"Retrying generation (attempt {retry_count + 1})")
                time.sleep(0.1)  # Brief pause between retries

    logger.error("All synthesis attempts failed")
    return None, None


# --- End File: engine.py ---
