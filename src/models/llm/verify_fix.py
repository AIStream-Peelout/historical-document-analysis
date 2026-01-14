
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.models.llm.structured_json_llm import StructuredJSONLLM

def test_structured_dir_in_result():
    # Mock the necessary components
    with patch('src.models.llm.structured_json_llm.StructuredJSONLLM._setup_lm_studio') as mock_setup, \
         patch('src.models.llm.structured_json_llm.Agent'), \
         patch('builtins.open', new_callable=MagicMock) as mock_open, \
         patch('json.load') as mock_json_load, \
         patch('pathlib.Path.mkdir'), \
         patch('pathlib.Path.exists', return_value=True):
        
        # Setup mock data
        mock_json_load.return_value = {
            'pdf_path': '/path/to/test.pdf',
            'pages': []
        }

        # Mock setup to set self.model
        def side_effect_setup(model_name, lm_studio_url):
             # We need to set the model attribute on the instance that calls this
             # But since this is a bound method call on the instance, we can't easily get 'self' here
             # unless we use autospec=True or similar, but simpler is to just mock Agent creation
             pass
        
        # Actually, let's just patch the Agent creation line or ensure self.model exists
        # Easier: Mock the whole __init__? No, we want to test process_ocr_results.
        
        # Let's just set self.model after init? No, it fails IN init.
        
        # We can patch the class method to set the attribute
        mock_setup.side_effect = lambda *args, **kwargs: None
        
        # We need to patch the Agent call to not need self.model, OR set self.model before Agent is called.
        # But Agent is called in __init__.
        
        # Let's use a subclass for testing that overrides __init__ or setup
        class TestLLM(StructuredJSONLLM):
            def _setup_lm_studio(self, *args):
                self.model = MagicMock()
                self.model_name = "test_model"

        # Initialize the service
        llm_service = TestLLM(raw_data_dir='/tmp/test_data')
        
        # Call the method
        result = llm_service.process_ocr_results_sync('/tmp/test_ocr.json')
        
        # Check if structured_dir is in the result
        if 'structured_dir' in result:
            print(f"SUCCESS: structured_dir found in result: {result['structured_dir']}")
        else:
            print("FAILURE: structured_dir NOT found in result")
            print(f"Result keys: {result.keys()}")

if __name__ == "__main__":
    test_structured_dir_in_result()
