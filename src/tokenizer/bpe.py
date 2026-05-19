from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class SimpleTokenizer:
    """A BPE Tokenizer implementing the same interface as the character-level tokenizer."""
    
    def __init__(self):
        # We use ByteLevel BPE tokenizer which is standard for language models (like GPT)
        # and guarantees that we never have out-of-vocabulary character issues.
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self.vocab_size = 0
        
    def train(self, text, vocab_size=2000):
        # Split text into lines to avoid feeding one massive string to the trainer
        iterator = text.split('\n')
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            vocab_size=vocab_size
        )
        self.tokenizer.train_from_iterator(iterator, trainer)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
    def encode(self, text):
        return self.tokenizer.encode(text).ids
        
    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)
        
    def save(self, filepath):
        self.tokenizer.save(filepath)
        
    def load(self, filepath):
        self.tokenizer = Tokenizer.from_file(filepath)
        self.vocab_size = self.tokenizer.get_vocab_size()

