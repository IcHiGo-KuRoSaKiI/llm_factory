"""
Dry-run logging utility for LLM Factory.
Handles saving dry-run requests to timestamped JSON files.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DryRunLogger:
    """
    Handles logging of dry-run requests to JSON files with timestamps.
    Creates organized logs for debugging and analysis.
    """
    
    def __init__(self, base_directory: str = None):
        """
        Initialize the dry-run logger.
        
        Args:
            base_directory: Base directory for logs. Defaults to current working directory.
        """
        if base_directory is None:
            base_directory = os.getcwd()
        
        self.dry_run_folder = os.path.join(base_directory, "llm_factory_dry_run_responses")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.request_counter = 0
        
        # Ensure the dry-run folder exists
        self._ensure_folder_exists()
        
        logger.info(f"DryRunLogger initialized with session ID: {self.session_id}")
        logger.info(f"Dry-run logs will be saved to: {self.dry_run_folder}")
    
    def _ensure_folder_exists(self):
        """Create the dry-run folder if it doesn't exist."""
        try:
            Path(self.dry_run_folder).mkdir(parents=True, exist_ok=True)
            logger.info(f"Dry-run folder ready: {self.dry_run_folder}")
        except Exception as e:
            logger.error(f"Failed to create dry-run folder {self.dry_run_folder}: {e}")
            # Fallback to current directory
            self.dry_run_folder = os.getcwd()
            logger.warning(f"Using fallback directory: {self.dry_run_folder}")
    
    def log_request(self, 
                   pipeline_name: str,
                   step_name: str,
                   step_type: str,
                   request_data: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log a dry-run request to a timestamped JSON file.
        
        Args:
            pipeline_name: Name of the pipeline
            step_name: Name of the step
            step_type: Type of the step
            request_data: The complete request data that would be sent to the API
            metadata: Additional metadata about the request
            
        Returns:
            Path to the created log file
        """
        self.request_counter += 1
        timestamp = datetime.now().isoformat()
        
        # Create comprehensive log entry
        log_entry = {
            "session_info": {
                "session_id": self.session_id,
                "request_number": self.request_counter,
                "timestamp": timestamp
            },
            "pipeline_info": {
                "pipeline_name": pipeline_name,
                "step_name": step_name,
                "step_type": step_type
            },
            "request_data": request_data,
            "metadata": metadata or {}
        }
        
        # Generate filename with timestamp and step info
        safe_pipeline_name = self._sanitize_filename(pipeline_name)
        safe_step_name = self._sanitize_filename(step_name)
        
        filename = f"{self.session_id}_{self.request_counter:03d}_{safe_pipeline_name}_{safe_step_name}_{step_type}.json"
        file_path = os.path.join(self.dry_run_folder, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Dry-run request logged to: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write dry-run log to {file_path}: {e}")
            return ""
    
    def log_pipeline_summary(self,
                           pipeline_name: str,
                           total_steps: int,
                           pipeline_config: Dict[str, Any],
                           execution_metadata: Dict[str, Any]) -> str:
        """
        Log a summary of the entire pipeline execution.
        
        Args:
            pipeline_name: Name of the pipeline
            total_steps: Total number of steps in the pipeline
            pipeline_config: Complete pipeline configuration
            execution_metadata: Metadata about the execution
            
        Returns:
            Path to the created summary file
        """
        timestamp = datetime.now().isoformat()
        
        summary_entry = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "summary_type": "pipeline_execution"
            },
            "pipeline_info": {
                "pipeline_name": pipeline_name,
                "total_steps": total_steps,
                "total_requests_logged": self.request_counter
            },
            "pipeline_config": pipeline_config,
            "execution_metadata": execution_metadata
        }
        
        safe_pipeline_name = self._sanitize_filename(pipeline_name)
        filename = f"{self.session_id}_SUMMARY_{safe_pipeline_name}.json"
        file_path = os.path.join(self.dry_run_folder, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary_entry, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Pipeline summary logged to: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write pipeline summary to {file_path}: {e}")
            return ""
    
    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a string to be safe for use in filenames.
        
        Args:
            name: String to sanitize
            
        Returns:
            Sanitized string safe for filenames
        """
        if not name:
            return "unnamed"
        
        # Replace problematic characters with underscores
        safe_chars = []
        for char in name:
            if char.isalnum() or char in '-_':
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        
        # Join and limit length
        safe_name = ''.join(safe_chars)
        return safe_name[:50]  # Limit to 50 characters
    
    def log_pipeline_result(self,
                          pipeline_name: str,
                          pipeline_result: Dict[str, Any],
                          pipeline_config: Dict[str, Any]) -> str:
        """
        Log the complete final result of a pipeline execution.
        This includes all conversation history, steps, results, etc.
        
        Args:
            pipeline_name: Name of the pipeline
            pipeline_result: Complete result object with all data
            pipeline_config: Original pipeline configuration
            
        Returns:
            Path to the created result file
        """
        timestamp = datetime.now().isoformat()
        
        result_entry = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "log_type": "complete_pipeline_result"
            },
            "pipeline_info": {
                "pipeline_name": pipeline_name,
                "total_requests_logged": self.request_counter
            },
            "pipeline_config": pipeline_config,
            "complete_result": pipeline_result
        }
        
        safe_pipeline_name = self._sanitize_filename(pipeline_name)
        filename = f"{self.session_id}_RESULT_{safe_pipeline_name}.json"
        file_path = os.path.join(self.dry_run_folder, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_entry, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Complete pipeline result logged to: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write pipeline result to {file_path}: {e}")
            return ""

    def log_fine_tuning_process(self,
                              pipeline_name: str,
                              step_name: str,
                              original_prompt: str,
                              fine_tuning_guidelines: str,
                              enhanced_prompt: str,
                              enhancement_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log the complete fine-tuning process showing before/after prompts.
        
        Args:
            pipeline_name: Name of the pipeline
            step_name: Name of the step being fine-tuned
            original_prompt: The original base prompt
            fine_tuning_guidelines: The fine-tuning instructions used
            enhanced_prompt: The resulting enhanced prompt
            enhancement_metadata: Additional metadata about the enhancement process
            
        Returns:
            Path to the created fine-tuning log file
        """
        timestamp = datetime.now().isoformat()
        
        fine_tuning_entry = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "log_type": "fine_tuning_process"
            },
            "pipeline_info": {
                "pipeline_name": pipeline_name,
                "step_name": step_name
            },
            "fine_tuning_process": {
                "original_prompt": original_prompt,
                "fine_tuning_guidelines": fine_tuning_guidelines,
                "enhanced_prompt": enhanced_prompt,
                "prompt_length_comparison": {
                    "original_length": len(original_prompt),
                    "enhanced_length": len(enhanced_prompt),
                    "length_increase": len(enhanced_prompt) - len(original_prompt)
                }
            },
            "enhancement_metadata": enhancement_metadata or {}
        }
        
        safe_pipeline_name = self._sanitize_filename(pipeline_name)
        safe_step_name = self._sanitize_filename(step_name)
        filename = f"{self.session_id}_FINETUNE_{safe_pipeline_name}_{safe_step_name}.json"
        file_path = os.path.join(self.dry_run_folder, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fine_tuning_entry, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Fine-tuning process logged to: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write fine-tuning log to {file_path}: {e}")
            return ""

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current logging session.
        
        Returns:
            Dictionary with session information
        """
        return {
            "session_id": self.session_id,
            "dry_run_folder": self.dry_run_folder,
            "requests_logged": self.request_counter,
            "folder_exists": os.path.exists(self.dry_run_folder)
        }


def create_dry_run_logger(base_directory: str = None) -> DryRunLogger:
    """
    Factory function to create a DryRunLogger instance.
    
    Args:
        base_directory: Base directory for logs. Defaults to current working directory.
        
    Returns:
        DryRunLogger instance
    """
    return DryRunLogger(base_directory)