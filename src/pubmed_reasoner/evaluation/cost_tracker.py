from dataclasses import dataclass
@dataclass
class CostTracker:
    input_tokens:int=0
    output_tokens:int=0
    input_rate:float|None=None
    output_rate:float|None=None
    def estimated_cost(self):
        if self.input_rate is None or self.output_rate is None: return None
        return self.input_tokens/1_000_000*self.input_rate+self.output_tokens/1_000_000*self.output_rate

