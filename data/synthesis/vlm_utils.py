import os
import cv2
import numpy as np
import sys
from pathlib import Path
import imageio
import io
import contextlib

# Try to import mediapy for better video codec support
try:
    import mediapy
    MEDIAPY_AVAILABLE = True
except ImportError:
    MEDIAPY_AVAILABLE = False

# Import LLM interface from llm folder
from llm.llm_interface import GetLLMInterface

# Verbosity control (will be set by the calling script)
VERBOSE = False

def vprint(*args, **kwargs):
    """Verbose print helper; prints only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)

def set_verbose(verbose: bool):
    """Set the verbosity level for this module."""
    global VERBOSE
    VERBOSE = verbose


def object_annotation(video_path: str, task_description: str, gpu_id: int = -1) -> dict:
    """
    Generate an object annotation for a video using VLM (uses first and last frames).
    
    :param video_path: Path to the input video file
    :param task_description: Task description from annotated npy files
    :return: Dictionary containing task description and object information
             Format: {"task": "Pick up cube", "object": "red cube"}
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file {video_path} does not exist")
        return {"task": "", "object": ""}
    
    vprint(f"Generating object annotation for {video_path.name}")
    
    # Extract first and last frames from the video
    try:
        reader = imageio.get_reader(str(video_path))
        
        # Get total number of frames
        try:
            total_frames = reader.count_frames()
        except AttributeError:
            # Fallback: count frames manually for legacy readers
            total_frames = 0
            for _ in reader:
                total_frames += 1
            reader.close()
            reader = imageio.get_reader(str(video_path))
        
        # Extract first and last frames and save them as temporary images
        if gpu_id == -1:
            temp_dir = Path(__file__).parent / "temp_annotation"
        else:
            temp_dir = Path(__file__).parent / f"temp_annotation_{gpu_id}"
        temp_dir.mkdir(exist_ok=True)
        
        temp_image_paths = []
        
        # First frame
        frame_rgb = reader.get_data(0)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        temp_image_path = temp_dir / "frame_0.jpg"
        cv2.imwrite(str(temp_image_path), frame_bgr)
        temp_image_paths.append(str(temp_image_path))
        
        # Last frame
        last_frame_idx = total_frames - 1 if total_frames > 1 else 0
        if last_frame_idx > 0:
            frame_rgb = reader.get_data(last_frame_idx)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            temp_image_path = temp_dir / f"frame_{last_frame_idx}.jpg"
            cv2.imwrite(str(temp_image_path), frame_bgr)
            temp_image_paths.append(str(temp_image_path))
            
        reader.close()
        
        vprint(f"Extracted {len(temp_image_paths)} frames for annotation (first and last)")
        
        # Initialize VLM interface
        try:
            llm_interface = GetLLMInterface(model_name="gpt-5.1", model_provider="Azure")
            vprint("Initialized VLM interface for object annotation")
            
            # LLM call: Object detection including related objects
            object_prompt = f"""You are analyzing a video where the task is: "{task_description}"

Identify relevant objects in the scene:
1. The PRIMARY object being manipulated, grasped, or moved (this is most important)
2. RELATED objects that are movable items on surfaces (other objects that could be picked up or moved)

CRITICAL EXCLUSIONS - DO NOT include:
- Robotic arms, robot hands, or any part of the robot
- Walls, floors, ceilings, or other fixed structural elements
- Large furniture like tables, desks, shelves (unless being moved)
- Background environmental elements

ONLY INCLUDE: Manipulable objects and items that could be interacted with or moved.

Describe each object using COLOR and OBJECT NAME format (e.g., "red cube", "blue towel", "green bottle").

Provide a period-separated list with the primary object FIRST, followed by related movable objects.

Examples of good responses:
- "red cube. blue bottle. yellow box"
- "blue bottle. green towel. black pen" 
- "black pen. white paper. red marker"
- "green cloth. blue box. yellow toy"

Task: {task_description}

Please provide a period-separated list of movable objects (primary object first, then related objects), nothing else. Exclude robots, walls, floors, and furniture."""
            
            object_messages = llm_interface.prepare_input_with_images(object_prompt, temp_image_paths)
            
            if not VERBOSE:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    object_response = llm_interface.get_string_response(object_messages)
            else:
                object_response = llm_interface.get_string_response(object_messages)
            
            vprint("Completed LLM call for object detection")
            
            # Clean up temporary files
            for temp_path in temp_image_paths:
                try:
                    os.remove(temp_path)
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
            
            # Process object response - now supports multiple objects
            object_result = object_response.strip()
            
            # Clean up object description
            if len(object_result) > 200:  # Increased limit for multiple objects
                object_result = object_result[:200].strip()
            
            # Remove quotes if present
            if object_result.startswith('"') and object_result.endswith('"'):
                object_result = object_result[1:-1]
            
            # Split into individual objects using period separator
            objects = [obj.strip() for obj in object_result.split('.') if obj.strip()]
            
            result = {
                "task": task_description,
                "object": objects[0] if objects else "",  # Primary object (first one)
                "related_objects": objects[1:] if len(objects) > 1 else [],  # Additional related objects
                "all_objects": object_result  # Full comma-separated string
            }
            
            vprint(f"Task description: {task_description}")
            vprint(f"Primary object: {objects[0] if objects else ''}")
            vprint(f"Related objects: {objects[1:] if len(objects) > 1 else []}")
            return result
            
        except Exception as e:
            print(f"Error with VLM interface: {e}")
            # Cleanup temporary files even if VLM fails
            for temp_path in temp_image_paths:
                try:
                    os.remove(temp_path)
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
            return {"task": task_description, "object": ""}
            
    except Exception as e:
        print(f"Error extracting frames from video: {e}")
        return {"task": task_description, "object": ""}


def task_rephrase(task_name: str, init_scene_text: str, video_path: str, gpu_id: int = -1) -> dict:
    """
    Generate instruction from task name, init scene, and video frames, then create rephrased version with sensor placeholder.
    
    Args:
        task_name: Name of the task (e.g., "Supermarket pickup - Produce area")
        init_scene_text: Initial scene description (e.g., "A robot is in front of the fruit shelf")
        video_path: Path to the video file for visual context
        gpu_id: GPU ID for temp file management (default: -1)
        
    Returns:
        dict: Dictionary with original and rephrased instructions
              Format: {"original_instruction": str, "rephrased_instruction": str}
    """
    try:
        vprint(f"Rephrasing task: {task_name}")
        vprint(f"Init scene: {init_scene_text}")
        vprint(f"Video path: {video_path}")
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"Error: Video file {video_path} does not exist")
            # Return fallback
            original_instruction = f"{task_name}. {init_scene_text}"
            return {
                "original_instruction": original_instruction,
                "rephrased_instruction": original_instruction
            }
        
        # Extract first and last frames from the video
        try:
            reader = imageio.get_reader(str(video_path))
            
            # Get total number of frames
            try:
                total_frames = reader.count_frames()
            except AttributeError:
                # Fallback: count frames manually for legacy readers
                total_frames = 0
                for _ in reader:
                    total_frames += 1
                reader.close()
                reader = imageio.get_reader(str(video_path))
            
            # Extract first and last frames and save them as temporary images
            if gpu_id == -1:
                temp_dir = Path(__file__).parent / "temp_rephrase"
            else:
                temp_dir = Path(__file__).parent / f"temp_rephrase_{gpu_id}"
            temp_dir.mkdir(exist_ok=True)
            
            temp_image_paths = []
            
            # First frame
            frame_rgb = reader.get_data(0)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            temp_image_path = temp_dir / "frame_0.jpg"
            cv2.imwrite(str(temp_image_path), frame_bgr)
            temp_image_paths.append(str(temp_image_path))
            
            # Last frame
            last_frame_idx = total_frames - 1 if total_frames > 1 else 0
            if last_frame_idx > 0:
                frame_rgb = reader.get_data(last_frame_idx)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                temp_image_path = temp_dir / f"frame_{last_frame_idx}.jpg"
                cv2.imwrite(str(temp_image_path), frame_bgr)
                temp_image_paths.append(str(temp_image_path))
                
            reader.close()
            
            vprint(f"Extracted {len(temp_image_paths)} frames for task rephrasing (first and last)")
            
        except Exception as e:
            print(f"Error extracting frames from video: {e}")
            # Return fallback
            original_instruction = f"{task_name}. {init_scene_text}"
            return {
                "original_instruction": original_instruction,
                "rephrased_instruction": original_instruction
            }
        
        # Initialize VLM interface
        try:
            llm_interface = GetLLMInterface(model_name="gpt-5.1", model_provider="Azure")
            vprint("Initialized VLM interface for task rephrasing")
            
            # Create prompt to generate both original and rephrased instructions
            rephrase_prompt = f"""You are analyzing video frames from a robot task. Generate two versions of the instruction:

1. ORIGINAL INSTRUCTION: Combine the task name, scene description, and visual information into a clear, concise robot instruction
2. REPHRASED INSTRUCTION: Same instruction but insert "<placeholder>" before the primary manipulable object

CRITICAL REQUIREMENTS:
- The original instruction should be natural and concise (combine task name, scene info, and what you see)
- The rephrased version should be identical except for inserting "<placeholder>" before the object
- The placeholder should be positioned as an adjective before the noun
- Only insert placeholder before objects that can be manipulated (grasped, picked, moved)
- DO NOT insert placeholder before robot parts, furniture, or fixed structures

Task Name: {task_name}
Initial Scene: {init_scene_text}

Examples of good outputs:
Task: "Pick up cube" / Scene: "Robot near table" 
→ Original: "Pick up cube from table"
→ Rephrased: "Pick up <placeholder> cube from table"

Task: "Supermarket pickup - Produce area" / Scene: "Robot in front of fruit shelf"
→ Original: "Pick up fruit from produce area"
→ Rephrased: "Pick up <placeholder> fruit from produce area"

Task: "Place bottle on shelf" / Scene: "Robot holding bottle"
→ Original: "Place bottle on shelf"
→ Rephrased: "Place <placeholder> bottle on shelf"

Please provide ONLY two lines in this exact format:
ORIGINAL: [your original instruction]
REPHRASED: [your rephrased instruction with <placeholder>]"""
            
            # Use the image interface with frames
            rephrase_messages = llm_interface.prepare_input_with_images(rephrase_prompt, temp_image_paths)
            
            if not VERBOSE:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    response = llm_interface.get_string_response(rephrase_messages)
            else:
                response = llm_interface.get_string_response(rephrase_messages)
            
            vprint("Completed LLM call for task rephrasing")
            
            # Clean up temporary files
            for temp_path in temp_image_paths:
                try:
                    os.remove(temp_path)
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
            
            # Parse the response
            response = response.strip()
            original_instruction = ""
            rephrased_instruction = ""
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith("ORIGINAL:"):
                    original_instruction = line.replace("ORIGINAL:", "").strip()
                elif line.startswith("REPHRASED:"):
                    rephrased_instruction = line.replace("REPHRASED:", "").strip()
            
            # Fallback if parsing failed
            if not original_instruction or not rephrased_instruction:
                vprint("Warning: Failed to parse LLM response, using fallback")
                original_instruction = f"{task_name}. {init_scene_text}"
                rephrased_instruction = original_instruction
            
            # Verify rephrased contains placeholder
            if "<placeholder>" not in rephrased_instruction.lower():
                vprint("Warning: Rephrased instruction missing <placeholder>, adding it")
                # Insert placeholder before last word (likely the object)
                words = rephrased_instruction.split()
                if len(words) >= 2:
                    rephrased_instruction = " ".join(words[:-1]) + " <placeholder> " + words[-1]
                else:
                    rephrased_instruction = f"<placeholder> {rephrased_instruction}"
            
            result = {
                "original_instruction": original_instruction,
                "rephrased_instruction": rephrased_instruction
            }
            
            vprint(f"Original: {original_instruction}")
            vprint(f"Rephrased: {rephrased_instruction}")
            
            return result
            
        except Exception as e:
            print(f"Error with VLM interface during task rephrasing: {e}")
            # Return fallback if VLM fails
            original_instruction = f"{task_name}. {init_scene_text}"
            return {
                "original_instruction": original_instruction,
                "rephrased_instruction": original_instruction
            }
        
    except Exception as e:
        print(f"Error rephrasing task: {e}")
        # Return original if rephrasing fails
        original_instruction = f"{task_name}. {init_scene_text}"
        return {
            "original_instruction": original_instruction,
            "rephrased_instruction": original_instruction
        }


def modify_task_with_sensor(text_list: list) -> list:
    """
    Modify a list of text descriptions by adding "<placeholder>" before manipulable objects.
    
    Args:
        text_list: List of strings containing task descriptions
        
    Returns:
        list: Modified list of strings with "<placeholder>" inserted before objects
    """
    try:
        vprint(f"Modifying text list to add <placeholder> before manipulable objects")
        vprint(f"Original text list: {text_list}")
        
        if not text_list or len(text_list) == 0:
            print("No valid text list provided")
            return []
        
        # Initialize VLM interface
        try:
            llm_interface = GetLLMInterface(model_name="gpt-5.1", model_provider="Azure")
            vprint("Initialized VLM interface for task modification")
            
            # Create text prompt to modify each description
            modification_prompt = f"""You are given a list of robot task descriptions. For each description, identify the manipulable object and insert "<placeholder>" immediately before it.

CRITICAL REQUIREMENTS:
1. Insert "<placeholder>" before the object that can be manipulated (grasped, picked, moved, etc.)
2. Keep the rest of the description unchanged
3. The placeholder should be positioned as an adjective before the noun
4. Don't change the original task too much - only insert the placeholder
5. Return ONLY a Python list format with the same number of items

Examples:
- "Pick up cube" -> "Pick up <placeholder> cube"
- "Place the bottle" -> "Place the <placeholder> bottle"
- "Open drawer" -> "Open <placeholder> drawer"
- "Grasp red ball" -> "Grasp <placeholder> red ball"
- "Move towel to table" -> "Move <placeholder> towel to table"

Original descriptions:
{text_list}

Please provide ONLY a Python list with "<placeholder>" inserted before manipulable objects, nothing else:"""
            
            # Use the text-only interface
            if not VERBOSE:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    modified_response = llm_interface.get_string_response([{"role": "user", "content": modification_prompt}])
            else:
                modified_response = llm_interface.get_string_response([{"role": "user", "content": modification_prompt}])
            
            vprint("Completed LLM call for task modification")
            
            # Process the modified response to extract the list
            modified_response = modified_response.strip()
            
            # Try to parse as a Python list
            try:
                # Remove any markdown code blocks if present
                if modified_response.startswith("```"):
                    lines = modified_response.split('\n')
                    modified_response = '\n'.join([l for l in lines if not l.startswith("```")])
                    modified_response = modified_response.strip()
                
                # Use eval to parse the list (safe since we control the LLM output)
                import ast
                modified_list = ast.literal_eval(modified_response)
                
                if not isinstance(modified_list, list):
                    raise ValueError("Response is not a list")
                
                # Ensure we have the same number of items
                if len(modified_list) != len(text_list):
                    vprint(f"Warning: Modified list has {len(modified_list)} items, original has {len(text_list)}")
                    # Pad or truncate to match
                    if len(modified_list) < len(text_list):
                        modified_list.extend(text_list[len(modified_list):])
                    else:
                        modified_list = modified_list[:len(text_list)]
                
                # Verify each item contains the placeholder
                for i, item in enumerate(modified_list):
                    if "<placeholder>" not in str(item).lower():
                        vprint(f"Warning: Item {i} doesn't contain '<placeholder>', forcing inclusion...")
                        # Try to intelligently insert the placeholder
                        words = str(item).split()
                        # Common patterns: "Pick [obj]", "Place [obj]", "Open [obj]", etc.
                        # Usually object comes after verb, so insert before last word(s)
                        if len(words) >= 2:
                            # Insert before the last word (likely the object)
                            modified_list[i] = " ".join(words[:-1]) + " <placeholder> " + words[-1]
                        else:
                            # If just one word, append placeholder
                            modified_list[i] = f"<placeholder> {item}"
                    
                
                vprint(f"Modified text list: {modified_list}")
                return modified_list
                
            except Exception as e:
                vprint(f"Error parsing LLM response as list: {e}")
                # Fallback: manually insert placeholder before last word (likely object)
                modified_list = []
                for text in text_list:
                    words = text.split()
                    if len(words) >= 2:
                        # Insert before the last word
                        modified_text = " ".join(words[:-1]) + " <placeholder> " + words[-1]
                    else:
                        # If just one word, prepend placeholder
                        modified_text = f"<placeholder> {text}"
                    
                    if len(modified_text) > 40:
                        modified_text = modified_text[:40].strip()
                    
                    modified_list.append(modified_text)
                
                vprint(f"Fallback modified text list: {modified_list}")
                return modified_list
            
        except Exception as e:
            print(f"Error with VLM interface during task modification: {e}")
            # Return original list if modification fails
            return text_list
            
    except Exception as e:
        print(f"Error modifying task list: {e}")
        return []
