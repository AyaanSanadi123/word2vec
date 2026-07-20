class LearningRateScheduler:
    def __init__(self,initial_lr:float,total_steps : int,min_lr = 0.0001):
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_rate(self,current_step : int) -> float:
        progress = current_step / self.total_steps # this gives u a number bw 0 and 1
        new_lr = self.initial_lr * (1.0 - progress) # linear decay formula 
        return max(new_lr,self.min_lr)
    
    