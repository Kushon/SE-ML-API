from unittest.mock import MagicMock, Mock

import torch

from ml import predict


def test_predict():
  
    device = torch.device('cpu')
    
    mock_tokenizer = Mock()
    mock_inputs = MagicMock()
    mock_inputs.__contains__ = lambda self, key: key in ['input_ids', 'attention_mask', 'token_type_ids']
    mock_inputs.__getitem__ = lambda self, key: {
        'input_ids': torch.ones(1, 128),
        'attention_mask': torch.ones(1, 128), 
        'token_type_ids': torch.zeros(1, 128)
    }[key]
    mock_inputs.to.return_value = mock_inputs  
    
    mock_tokenizer.return_value = mock_inputs
    
    
    mock_outputs = {'logits': torch.tensor([0.6])}  
    mock_model = Mock(return_value=mock_outputs)
    
 
    texts = ["test text 1", "test text 2"]
    predictions = predict(mock_tokenizer, mock_model, device, texts)
    
    assert predictions == [1, 1]
    mock_tokenizer.assert_any_call("test text 1", return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    mock_tokenizer.assert_any_call("test text 2", return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    mock_inputs.to.assert_called_with(device)
    
    
    
