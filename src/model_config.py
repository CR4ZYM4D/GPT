class ModelBlockConfig():


    def __init__(self, 
                 embedding_dimension: int = 14, 
                 batch_size: int = 4,
                 max_sequence_length: int = 1280, 
                 num_heads: int = 14, 
                 dropout_fraction: float = 0.4):
        

        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.dropout_fraction = dropout_fraction


class ModelConfig():

    def __init__(self, config: ModelBlockConfig, tokenizer, num_layers: int = 8):
        
        self.num_layers = num_layers
        
        self.block_config = config

        self.tokenizer = tokenizer
    
        self.vocab_size = len(self.tokenizer)