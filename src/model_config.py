class ModelBlockConfig():


    def __init__(self, 
                 embedding_dimension: int = 1024, 
                 batch_size: int = 512,
                 max_sequence_length: int = 1024, 
                 num_heads: int = 8, 
                 dropout_fraction: float = 0.4):
        

        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.dropout_fraction = dropout_fraction


# the number of parameters in this config is slightly more than that of the number of parameters 
# in the GPT-1 Model and GPT-2 small i.e ~133M parameters in this config v/s the ~119M parameters in the GPT-1 model and ~117M in the GPT-2 small model
class GPT1ModelConfig():

    def __init__(self, config: ModelBlockConfig, tokenizer, num_layers: int = 8, temperature: float = 0.7):
        
        self.num_layers = num_layers
        
        self.block_config = config

        self.tokenizer = tokenizer

        self.temperature = temperature
    
        self.vocab_size = len(self.tokenizer)

# the number of parameters in this config is slightly less than that of the number 
# of parameters in the GPT-2 Model i.e ~1.3B parameters in this config v/s the ~1.4B parameters in the GPT-2 model
class GPT2ModelConfig():

    def __init__(self, config: ModelBlockConfig, tokenizer, num_layers: int = 128, temperature: float = 0.7):
        
        self.num_layers = num_layers
        
        self.temperature = temperature

        self.block_config = config

        self.tokenizer = tokenizer
    
        self.vocab_size = len(self.tokenizer)      