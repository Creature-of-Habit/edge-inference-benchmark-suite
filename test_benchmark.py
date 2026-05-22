import pytest
import onnxruntime as ort

# Import the function you want to test from your main script
from benchmark import load_session 

def test_load_session_with_fake_file():
    """
    DISASTER PATH: Ensure the engine crashes when given an invalid file path.
    """
    fake_path = "models/fake_ghost_model.onnx"
    
    # 1. Define the 'blast zone' and tell pytest we expect an Exception
    with pytest.raises(Exception) as error_info:
        # 2. Run the code that should trigger the crash
        load_session(fake_path)
        
    # 3. (Optional but professional) Prove the error message mentions the fake file
    assert fake_path in str(error_info.value)