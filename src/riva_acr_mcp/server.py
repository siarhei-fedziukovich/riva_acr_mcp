#!/usr/bin/env python3
"""
NVIDIA Riva ACR MCP Server

A Model Context Protocol server that provides speech-to-text functionality using NVIDIA Riva.
Uses HTTP streamable transport for communication.
"""

import argparse
import os
import tempfile
import requests
from urllib.parse import urlparse
# import subprocess  # Removed - no longer using external processes
from pathlib import Path
from typing import Optional
import traceback
import logging
import sys
import wave
import audioop
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('riva_acr_mcp.log')
    ]
)
logger = logging.getLogger(__name__)


# Create the MCP server instance
mcp = FastMCP("NVIDIA Riva ACR MCP Server")


class AuthConfig:
    """Configuration for authentication credentials."""
    
    def __init__(self):
        self.credentials = {}
        self._load_from_env()
    
    def _load_from_env(self):
        """Load authentication credentials from environment variables."""
        # Format: AUTH_<IDENTIFIER>=base_url|username:password
        # Example: AUTH_MYSERVICE=https://api.example.com|user:pass
        for key, value in os.environ.items():
            if key.startswith('AUTH_'):
                if '|' in value:
                    base_url_part, auth_part = value.split('|', 1)
                    if ':' in auth_part:
                        username, password = auth_part.split(':', 1)
                        
                        # Parse the base URL to get the netloc for matching
                        parsed_base = urlparse(base_url_part)
                        if parsed_base.netloc:
                            self.credentials[parsed_base.netloc.lower()] = {
                                'username': username,
                                'password': password,
                                'type': 'basic',
                                'base_url': base_url_part
                            }
    
    def get_auth_for_url(self, url: str) -> Optional[tuple]:
        """Get authentication credentials for a given URL."""
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        
        if netloc in self.credentials:
            cred = self.credentials[netloc]
            if cred['type'] == 'basic':
                return (cred['username'], cred['password'])
        
        return None


# Global auth configuration
auth_config = AuthConfig()


def get_file_extension_from_content_type(content_type: str) -> str:
    """
    Get appropriate file extension from HTTP Content-Type header.
    
    Args:
        content_type: HTTP Content-Type header value
        
    Returns:
        str: File extension (with dot) or empty string if unknown
    """
    content_type = content_type.lower().split(';')[0].strip()  # Remove charset info
    
    extension_map = {
        'audio/wav': '.wav',
        'audio/x-wav': '.wav',
        'audio/wave': '.wav',
        'audio/mpeg': '.mp3',
        'audio/mp3': '.mp3',
        'audio/mp4': '.m4a',
        'audio/m4a': '.m4a',
        'audio/aac': '.aac',
        'audio/ogg': '.ogg',
        'audio/flac': '.flac',
        'audio/webm': '.webm',
        'audio/3gpp': '.3gp',
        'audio/amr': '.amr',
    }
    
    return extension_map.get(content_type, '')


def download_audio(url: str, output_base_path: str) -> tuple[bool, str, str]:
    """
    Download audio file from URL with authentication support and proper file extension.
    
    Args:
        url: URL to download from
        output_base_path: Base path for the output file (extension will be added based on content type)
        
    Returns:
        tuple[bool, str, str]: (Success status, Error message if failed, Final file path)
    """
    try:
        logger.info(f"Starting download from URL: {url}")
        logger.info(f"Base output path: {output_base_path}")
        
        # Validate output directory exists
        output_dir = Path(output_base_path).parent
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return False, f"Output directory does not exist: {output_dir}", ""
        
        # Get authentication for this URL if available
        auth = auth_config.get_auth_for_url(url)
        if auth:
            logger.info(f"Using authentication for URL: {url}")
        
        # Download with authentication if available
        logger.info(f"Initiating HTTP request to: {url}")
        response = requests.get(url, auth=auth, stream=True, timeout=30)
        response.raise_for_status()
        
        logger.info(f"HTTP response status: {response.status_code}")
        content_type = response.headers.get('content-type', 'unknown')
        logger.info(f"Content type: {content_type}")
        logger.info(f"Content length: {response.headers.get('content-length', 'unknown')}")
        
        # Determine file extension from content type
        file_extension = get_file_extension_from_content_type(content_type)
        final_output_path = output_base_path + file_extension
        
        logger.info(f"Determined file extension: '{file_extension}' from content type: {content_type}")
        logger.info(f"Final output path: {final_output_path}")
        
        bytes_written = 0
        with open(final_output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
        
        # Verify file was created and has content
        if not Path(final_output_path).exists():
            error_msg = f"Downloaded file was not created at: {final_output_path}"
            logger.error(error_msg)
            return False, error_msg, ""
        
        file_size = Path(final_output_path).stat().st_size
        logger.info(f"Successfully downloaded {bytes_written} bytes to: {final_output_path}")
        logger.info(f"Final file size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = f"Downloaded file is empty: {final_output_path}"
            logger.error(error_msg)
            return False, error_msg, ""
        
        return True, "", final_output_path
        
    except requests.exceptions.RequestException as e:
        error_msg = f"HTTP request failed for {url}: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, error_msg, ""
    except IOError as e:
        error_msg = f"File I/O error when writing to {output_base_path}: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, error_msg, ""
    except Exception as e:
        error_msg = f"Unexpected error downloading audio from {url} to {output_base_path}: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, error_msg, ""


def load_wav_with_builtin(wav_path: str):
    """
    Load WAV file using Python's built-in libraries and return a pydub-compatible object.
    
    Args:
        wav_path: Path to the WAV file
        
    Returns:
        AudioSegment-like object with channels, frame_rate, sample_width, and raw_data
    """
    logger.info(f"Loading WAV file with built-in libraries: {wav_path}")
    
    with wave.open(wav_path, 'rb') as wav_file:
        # Get WAV file properties
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        
        logger.info(f"WAV properties - Channels: {channels}, Sample width: {sample_width}, Frame rate: {frame_rate}, Frames: {frames}")
        
        # Read all audio data
        raw_data = wav_file.readframes(frames)
        
    # Create a simple object that mimics pydub's AudioSegment interface
    class SimpleAudioSegment:
        def __init__(self, raw_data, channels, frame_rate, sample_width):
            self._raw_data = raw_data
            self.channels = channels
            self.frame_rate = frame_rate
            self.sample_width = sample_width
            self._duration_ms = (len(raw_data) / (channels * sample_width * frame_rate)) * 1000
        
        def __len__(self):
            return int(self._duration_ms)
        
        def set_channels(self, new_channels):
            if new_channels == self.channels:
                return self
            
            if self.channels == 2 and new_channels == 1:
                # Convert stereo to mono
                logger.info("Converting stereo to mono using built-in audioop")
                mono_data = audioop.tomono(self._raw_data, self.sample_width, 1, 1)
                return SimpleAudioSegment(mono_data, 1, self.frame_rate, self.sample_width)
            else:
                raise ValueError(f"Unsupported channel conversion: {self.channels} -> {new_channels}")
        
        def set_frame_rate(self, new_rate):
            if new_rate == self.frame_rate:
                return self
            
            logger.info(f"Converting sample rate from {self.frame_rate}Hz to {new_rate}Hz using built-in audioop")
            # Use audioop to change sample rate
            converted_data, _ = audioop.ratecv(
                self._raw_data, self.sample_width, self.channels, 
                self.frame_rate, new_rate, None
            )
            return SimpleAudioSegment(converted_data, self.channels, new_rate, self.sample_width)
        
        def set_sample_width(self, new_width):
            if new_width == self.sample_width:
                return self
            
            logger.info(f"Converting sample width from {self.sample_width} bytes to {new_width} bytes using built-in audioop")
            # Convert sample width using audioop
            if self.sample_width == 1 and new_width == 2:
                converted_data = audioop.lin2lin(self._raw_data, 1, 2)
            elif self.sample_width == 2 and new_width == 1:
                converted_data = audioop.lin2lin(self._raw_data, 2, 1)
            elif self.sample_width == 2 and new_width == 4:
                converted_data = audioop.lin2lin(self._raw_data, 2, 4)
            elif self.sample_width == 4 and new_width == 2:
                converted_data = audioop.lin2lin(self._raw_data, 4, 2)
            else:
                raise ValueError(f"Unsupported sample width conversion: {self.sample_width} -> {new_width}")
            
            return SimpleAudioSegment(converted_data, self.channels, self.frame_rate, new_width)
        
        def export(self, output_path, format="wav"):
            if format != "wav":
                raise ValueError(f"Built-in WAV processor only supports WAV export, not {format}")
            
            logger.info(f"Exporting WAV file to: {output_path}")
            with wave.open(output_path, 'wb') as out_wav:
                out_wav.setnchannels(self.channels)
                out_wav.setsampwidth(self.sample_width)
                out_wav.setframerate(self.frame_rate)
                out_wav.writeframes(self._raw_data)
    
    return SimpleAudioSegment(raw_data, channels, frame_rate, sample_width) # type: ignore


def convert_to_mono_wav(input_path: str, output_path: str) -> tuple[bool, str]:
    """
    Convert audio file to mono-channel WAV format using pydub (pure Python).
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file
        
    Returns:
        tuple[bool, str]: (Success status, Error message if failed)
    """
    try:
        logger.info(f"Starting audio conversion from: {input_path}")
        logger.info(f"Target output path: {output_path}")
        
        # Verify input file exists and is readable
        input_file = Path(input_path)
        if not input_file.exists():
            error_msg = f"Input file does not exist: {input_path}"
            logger.error(error_msg)
            return False, error_msg
        
        if not input_file.is_file():
            error_msg = f"Input path is not a file: {input_path}"
            logger.error(error_msg)
            return False, error_msg
        
        input_size = input_file.stat().st_size
        logger.info(f"Input file size: {input_size} bytes")
        
        if input_size == 0:
            error_msg = f"Input file is empty: {input_path}"
            logger.error(error_msg)
            return False, error_msg
        
        # Validate output directory exists
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return False, f"Output directory does not exist: {output_dir}"
        
        # Import pydub for audio processing
        try:
            from pydub import AudioSegment
            logger.info("Successfully imported pydub")
        except ImportError as e:
            error_msg = f"Failed to import pydub: {e}. Please install with: pip install pydub"
            logger.error(error_msg)
            return False, error_msg
        
        # Load audio file (pydub can handle many formats)
        logger.info(f"Loading audio file: {input_path}")
        try:
            # First try to load with pydub
            audio = AudioSegment.from_file(input_path)
            logger.info(f"Successfully loaded audio file with pydub")
            logger.info(f"Original format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
        except Exception as e:
            logger.warning(f"Pydub failed to load file (possibly missing ffmpeg): {e}")
            
            # Try fallback for WAV files using built-in libraries
            if input_path.lower().endswith('.wav'):
                logger.info("Attempting fallback WAV processing with built-in libraries...")
                try:
                    audio = load_wav_with_builtin(input_path)
                    logger.info(f"Successfully loaded WAV file with built-in libraries")
                    logger.info(f"Original format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
                except Exception as fallback_e:
                    error_msg = f"Both pydub and built-in WAV processing failed. Pydub error: {e}. Built-in error: {fallback_e}"
                    logger.error(error_msg)
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    return False, error_msg
            else:
                error_msg = f"Failed to load audio file {input_path}: {e}. For non-WAV files, ffmpeg is required for pydub."
                logger.error(error_msg)
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False, error_msg
        
        # Convert to mono
        if audio.channels > 1:
            logger.info(f"Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
        else:
            logger.info("Audio is already mono")
        
        # Set sample rate to 16kHz
        if audio.frame_rate != 16000:
            logger.info(f"Converting sample rate from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)
        else:
            logger.info("Audio is already at 16kHz")
        
        # Set sample width to 16-bit (2 bytes)
        if audio.sample_width != 2:
            logger.info(f"Converting sample width from {audio.sample_width} bytes to 2 bytes (16-bit)")
            audio = audio.set_sample_width(2)
        else:
            logger.info("Audio is already 16-bit")
        
        logger.info(f"Final format - Channels: {audio.channels}, Frame rate: {audio.frame_rate}, Sample width: {audio.sample_width}, Duration: {len(audio)}ms")
        
        # Export as WAV
        logger.info(f"Exporting WAV file to: {output_path}")
        audio.export(output_path, format="wav")
        
        # Verify output file was created
        output_file = Path(output_path)
        if not output_file.exists():
            error_msg = f"Output file was not created: {output_path}"
            logger.error(error_msg)
            return False, error_msg
        
        output_size = output_file.stat().st_size
        logger.info(f"Successfully created WAV file: {output_path}")
        logger.info(f"Output file size: {output_size} bytes")
        
        if output_size == 0:
            error_msg = f"Output file is empty: {output_path}"
            logger.error(error_msg)
            return False, error_msg
        
        return True, ""
        
    except ImportError as e:
        error_msg = f"Import error during audio conversion: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during audio conversion from {input_path} to {output_path}: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False, error_msg


def transcribe_with_riva_offline(asr_service, config, audio_data: bytes) -> str:
    """
    Perform offline speech recognition using NVIDIA Riva.
    
    Args:
        asr_service: Initialized Riva ASR service
        config: Riva recognition config
        audio_data: Raw audio data bytes
        
    Returns:
        str: Transcribed text or error message
    """
    try:
        logger.info("Performing offline speech recognition...")
        response = asr_service.offline_recognize(audio_data, config) # type: ignore
        logger.info("Successfully completed offline speech recognition")
        
        # Extract transcript from all response results, using most probable alternatives
        if response.results:
            logger.info(f"Processing {len(response.results)} result segments from offline recognition")
            
            # Build transcript from all results, selecting best alternative for each
            transcript_segments = []
            confidence_scores = []
            total_alternatives_processed = 0
            
            for i, result in enumerate(response.results):
                if result.alternatives:
                    # Find the alternative with highest confidence
                    best_alternative = max(result.alternatives, key=lambda alt: alt.confidence)
                    
                    logger.debug(f"Segment {i+1}: {len(result.alternatives)} alternatives, "
                               f"best confidence: {best_alternative.confidence:.3f}")
                    
                    transcript_segments.append(best_alternative.transcript)
                    confidence_scores.append(best_alternative.confidence)
                    total_alternatives_processed += len(result.alternatives)
                    
                    # Log all alternatives for debugging
                    for j, alt in enumerate(result.alternatives):
                        logger.debug(f"  Alternative {j+1}: '{alt.transcript}' (confidence: {alt.confidence:.3f})")
                else:
                    logger.warning(f"Segment {i+1} has no alternatives")
            
            if transcript_segments:
                # Combine all segments into final transcript
                final_transcript = " ".join(transcript_segments).strip()
                
                # Calculate overall confidence (weighted average by segment length)
                if confidence_scores:
                    segment_lengths = [len(seg) for seg in transcript_segments]
                    total_length = sum(segment_lengths)
                    
                    if total_length > 0:
                        weighted_confidence = sum(
                            conf * length / total_length 
                            for conf, length in zip(confidence_scores, segment_lengths)
                        )
                    else:
                        weighted_confidence = sum(confidence_scores) / len(confidence_scores)
                else:
                    weighted_confidence = 0.0
                
                logger.info(f"Offline transcription successful:")
                logger.info(f"  - Segments processed: {len(transcript_segments)}")
                logger.info(f"  - Total alternatives evaluated: {total_alternatives_processed}")
                logger.info(f"  - Final transcript length: {len(final_transcript)} characters")
                logger.info(f"  - Weighted confidence: {weighted_confidence:.3f}")
                logger.info(f"  - Individual segment confidences: {[f'{c:.3f}' for c in confidence_scores]}")
                
                return f"Transcript: {final_transcript} (Confidence: {weighted_confidence:.2f})"
            else:
                logger.warning("No valid transcript segments found in offline recognition results")
                return "No speech detected in the audio file"
        else:
            logger.warning("No results returned from offline recognition")
            return "No speech detected in the audio file"
            
    except Exception as e:
        error_msg = f"Failed during offline speech recognition: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error: {error_msg}"


def transcribe_with_riva_streaming(asr_service, config, audio_data: bytes) -> str:
    """
    Perform streaming speech recognition using NVIDIA Riva.
    
    Args:
        asr_service: Initialized Riva ASR service
        config: Riva recognition config
        audio_data: Raw audio data bytes
        
    Returns:
        str: Transcribed text or error message
    """
    try:
        logger.info("Performing streaming speech recognition...")
        
        # Import riva.client within the function to ensure it's available
        import riva.client
        
        # Configure streaming settings
        streaming_config = riva.client.StreamingRecognitionConfig(
            config=config,
            interim_results=True  # Get partial results during streaming
        )
        
        # Create a generator to yield audio chunks
        def audio_chunks_generator():
            chunk_size = 1024 * 16  # 16KB chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if chunk:
                    logger.debug(f"Yielding audio chunk {i//chunk_size + 1}, size: {len(chunk)} bytes")
                    # Yield raw bytes, not wrapped in StreamingRecognizeRequest
                    # The streaming_response_generator handles the request wrapping
                    yield chunk
        
        # Perform streaming recognition
        logger.info(f"Starting streaming recognition with {len(audio_data)} bytes of audio data")
        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_chunks_generator(),
            streaming_config=streaming_config
        )
        
        # Collect results from streaming responses
        final_transcript = ""
        best_confidence = 0.0
        partial_results = []
        
        for response in responses:
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        confidence = result.alternatives[0].confidence
                        
                        if result.is_final:
                            logger.info(f"Final result: '{transcript}' (confidence: {confidence:.2f})")
                            final_transcript += transcript + " "
                            if confidence > best_confidence:
                                best_confidence = confidence
                        else:
                            logger.debug(f"Interim result: '{transcript}' (confidence: {confidence:.2f})")
                            partial_results.append((transcript, confidence))
        
        # Clean up final transcript
        final_transcript = final_transcript.strip()
        
        if final_transcript:
            logger.info(f"Streaming transcription successful - Best confidence: {best_confidence:.2f}")
            logger.info(f"Final transcript length: {len(final_transcript)} characters")
            logger.info(f"Processed {len(partial_results)} interim results")
            
            return f"Transcript: {final_transcript} (Confidence: {best_confidence:.2f})"
        else:
            logger.warning("No speech detected in the audio file (streaming mode)")
            if partial_results:
                # If we have partial results but no final, return the best partial result
                best_partial = max(partial_results, key=lambda x: x[1])
                logger.info(f"Using best partial result: '{best_partial[0]}' (confidence: {best_partial[1]:.2f})")
                return f"Transcript (partial): {best_partial[0]} (Confidence: {best_partial[1]:.2f})"
            else:
                return "No speech detected in the audio file"
                
    except Exception as e:
        error_msg = f"Failed during streaming speech recognition: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error: {error_msg}"


def transcribe_with_riva(audio_path: str) -> str:
    """
    Transcribe audio using NVIDIA Riva ASR service.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        str: Transcribed text or error message
    """
    try:
        logger.info(f"Starting NVIDIA Riva transcription for: {audio_path}")
        
        # Verify audio file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            error_msg = f"Audio file does not exist for transcription: {audio_path}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        audio_size = audio_file.stat().st_size
        logger.info(f"Audio file size: {audio_size} bytes")
        
        if audio_size == 0:
            error_msg = f"Audio file is empty: {audio_path}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        # Import NVIDIA Riva client
        try:
            import riva.client
            logger.info("Successfully imported NVIDIA Riva client")
        except ImportError as e:
            error_msg = f"Failed to import nvidia-riva-client: {e}. Please install it with: pip install nvidia-riva-client"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        # Get Riva server URI from environment variable or use default
        riva_uri = os.getenv('RIVA_URI', 'localhost:50051')
        logger.info(f"Using Riva server URI: {riva_uri}")
        
        # Get Riva ASR mode from environment variable (offline or streaming)
        riva_mode = os.getenv('RIVA_ASR_MODE', 'offline').lower()
        if riva_mode not in ['offline', 'streaming']:
            logger.warning(f"Invalid RIVA_ASR_MODE '{riva_mode}', defaulting to 'offline'")
            riva_mode = 'offline'
        logger.info(f"Using Riva ASR mode: {riva_mode}")
        
        # Get number of alternatives to request
        max_alternatives = int(os.getenv('RIVA_MAX_ALTERNATIVES', '3'))
        if max_alternatives < 1:
            logger.warning(f"Invalid RIVA_MAX_ALTERNATIVES '{max_alternatives}', defaulting to 3")
            max_alternatives = 3
        elif max_alternatives > 10:
            logger.warning(f"RIVA_MAX_ALTERNATIVES '{max_alternatives}' is high, capping at 10")
            max_alternatives = 10
        logger.info(f"Requesting {max_alternatives} alternatives per segment")
        
        # Initialize Riva client
        try:
            logger.info("Initializing Riva authentication...")
            auth = riva.client.Auth(uri=riva_uri)
            logger.info("Creating ASR service...")
            asr_service = riva.client.ASRService(auth)
            logger.info("Successfully initialized Riva client")
        except Exception as e:
            error_msg = f"Failed to initialize Riva client with URI {riva_uri}: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error: {error_msg}"
        
        # Configure recognition settings
        try:
            logger.info("Configuring recognition settings...")
            config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code="en-US",  # Set language code to match server models
                max_alternatives=max_alternatives,  # Request multiple alternatives to choose the best one
                enable_automatic_punctuation=True,
                verbatim_transcripts=False,
            )
            
            # Add audio file specifications to config
            logger.info(f"Adding audio file specs to config for: {audio_path}")
            riva.client.add_audio_file_specs_to_config(config, audio_path)
            logger.info(f"Recognition config - Encoding: {config.encoding}, Sample rate: {config.sample_rate_hertz}, Channels: {config.audio_channel_count}")
        except Exception as e:
            error_msg = f"Failed to configure recognition settings: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error: {error_msg}"
        
        # Read audio file
        try:
            logger.info(f"Reading audio file: {audio_path}")
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            logger.info(f"Successfully read {len(audio_data)} bytes of audio data")
        except IOError as e:
            error_msg = f"Failed to read audio file {audio_path}: {e}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error: {error_msg}"
        
        # Choose ASR method based on mode
        if riva_mode == 'streaming':
            logger.info("Using streaming ASR mode")
            return transcribe_with_riva_streaming(asr_service, config, audio_data)
        else:
            logger.info("Using offline ASR mode")
            return transcribe_with_riva_offline(asr_service, config, audio_data)
        
    except Exception as e:
        error_msg = f"Unexpected error during NVIDIA Riva transcription: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error: {error_msg}"


@mcp.tool()
def speech_to_text(url_2_recording: str) -> str:
    """
    Convert speech from an audio recording to text using NVIDIA Riva ASR.
    
    This function performs the following steps:
    1. Download audio from the provided URL (with auth support)
    2. Convert to mono-channel WAV format
    3. Send to NVIDIA Riva for speech recognition
    
    Args:
        url_2_recording (str): HTTP/HTTPS URL to the audio recording file
        
    Returns:
        str: Transcribed text from the audio recording
    """
    try:
        logger.info(f"Starting speech-to-text processing for URL: {url_2_recording}")
        
        # Validate URL format
        if not url_2_recording.startswith(('http://', 'https://')):
            error_msg = "Error: Only HTTP and HTTPS URLs are supported"
            logger.error(f"{error_msg}. Received URL: {url_2_recording}")
            return error_msg
        
        # Validate URL structure
        try:
            parsed_url = urlparse(url_2_recording)
            if not parsed_url.netloc:
                error_msg = f"Error: Invalid URL format: {url_2_recording}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Error: Failed to parse URL {url_2_recording}: {e}"
            logger.error(error_msg)
            return error_msg
        
        # Create temporary files
        logger.info("Creating temporary directory for processing...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            logger.info(f"Using temporary directory: {temp_dir_path}")
            
            # Step 1: Download audio file
            original_file_base = temp_dir_path / "original_audio"
            logger.info(f"Step 1: Downloading audio to: {original_file_base}")
            
            success, error_msg, final_downloaded_path = download_audio(url_2_recording, str(original_file_base))
            if not success:
                logger.error(f"Download failed: {error_msg}")
                return f"Error: Failed to download audio from {url_2_recording}. Details: {error_msg}"
            
            logger.info(f"Step 1 completed: Audio download successful to {final_downloaded_path}")
            
            # Step 2: Convert to mono WAV
            wav_file = temp_dir_path / "converted_audio.wav"
            logger.info(f"Step 2: Converting audio to mono WAV: {wav_file}")
            
            success, error_msg = convert_to_mono_wav(final_downloaded_path, str(wav_file))
            if not success:
                logger.error(f"Audio conversion failed: {error_msg}")
                return f"Error: Failed to convert audio to mono WAV format. Details: {error_msg}"
            
            logger.info("Step 2 completed: Audio conversion successful")
            
            # Step 3: Transcribe with NVIDIA Riva
            logger.info("Step 3: Starting transcription with NVIDIA Riva")
            transcription = transcribe_with_riva(str(wav_file))
            
            if transcription.startswith("Error:"):
                logger.error(f"Transcription failed: {transcription}")
            else:
                logger.info("Step 3 completed: Transcription successful")
            
            return transcription
            
    except Exception as e:
        error_msg = f"Unexpected error during speech-to-text processing: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error: {error_msg}"


# Add health check endpoint for Docker and monitoring
def setup_health_endpoint():
    """Set up health check endpoint for the FastAPI app."""
    try:
        # Get the FastAPI app instance
        app = mcp.streamable_http_app()
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint for Docker and monitoring systems."""
            from datetime import datetime
            return {
                "status": "healthy",
                "service": "NVIDIA Riva ACR MCP Server",
                "version": "0.1.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": ["speech_to_text"]
            }
        
        logger.info("Health check endpoint configured at /health")
        return app
    except Exception as e:
        logger.warning(f"Could not set up health endpoint: {e}")
        return None


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="NVIDIA Riva ACR MCP Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host to bind the server to (default: localhost)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting NVIDIA Riva ACR MCP Server on http://{args.host}:{args.port}")
    print(f"MCP endpoint will be available at: http://{args.host}:{args.port}/mcp")
    
    # Run the server with streamable HTTP transport
    # FastMCP.run() uses uvicorn internally, but we need to pass host/port via uvicorn directly
    import uvicorn
    
    # Set up health endpoint and get the app
    app = setup_health_endpoint()
    if app is None:
        # Fallback to basic app if health endpoint setup failed
        app = mcp.streamable_http_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
