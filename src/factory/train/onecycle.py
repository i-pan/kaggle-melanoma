try:
    from torch.optim.lr_scheduler import OneCycleLR
except:
    from .torch_lr_scheduler import OneCycleLR

class CustomOneCycleLR(OneCycleLR):
    '''
    This implements a 2-phase one cycle learning rate policy 
    that fast.ai says improves performance.

    Phase 1: Increase LR linearly from initial to max
    Phase 2: Decrease LR using cosine annealing from max to 0
    '''
    def get_lr(self):

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                # Linear on the way up
                computed_lr = self._annealing_linear(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self._annealing_linear(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                # Cosine on the way down
                down_step_num = step_num - self.step_size_up
                computed_lr = self._annealing_cos(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self._annealing_cos(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs