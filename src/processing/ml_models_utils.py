
"""@package ml_models_utils.py
Classes with utility functions for machine learning pytorch/skorch models.
"""
import torch
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from skorch.callbacks import Callback, Checkpoint, EarlyStopping, LoadInitState, LRScheduler
from src.utl.logging import Log as log



import torch
from skorch.callbacks import Callback


class CustomGradientClipping(Callback):
    def __init__(self, clip_value=None):
        self.clip_value = clip_value

    def on_grad_computed(self, net, named_parameters, **kwargs):
        # Se clip_value for None, o grid search ignora o clipping
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in named_parameters if p.grad is not None], 
                max_norm=self.clip_value
            )


# 2. Esta é a nova classe que herda da classe base e adiciona o monitoramento
class MonitoredGradClip(CustomGradientClipping):
    def __init__(self, clip_value=1.0, print_every=5, iterations=0):
        # Inicializa a classe base herdando o clip_value
        super().__init__(clip_value=clip_value)
        self.print_every = print_every
        self.iterations = iterations  # Contador de lotes (batches)

    def on_grad_computed(self, net, named_parameters, **kwargs):

        # Executa o print apenas no intervalo definido (padrão: a cada 50 batches)
        should_print = (self.iterations % self.print_every == 0)

        if should_print:
            print(f"\n[AUDITORIA CLIP] Batch #{self.iterations} | clip_value is {self.clip_value}")

        if self.clip_value is None:
            return

        # Filtra os parâmetros com gradiente válido
        params = [p for _, p in named_parameters if p.grad is not None]
        if not params:
            return

        if should_print:
            # Calcula a norma L2 total antes da modificação do PyTorch
            norma_antes = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
            ).item()

        # Chama a lógica de clipping da classe base que criamos anteriormente
        super().on_grad_computed(net, named_parameters, **kwargs)

        if should_print:
            # Calcula a norma L2 total após a execução do clipping
            norma_depois = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
            ).item()
            
            print(f"\n[AUDITORIA CLIP] Batch #{self.iterations} | "
                  f"Norma Original: {norma_antes:.4f} | "
                  f"Norma Pós-Clip (Limite: {self.clip_value}): {norma_depois:.4f}")

        # Incrementa o contador de iterações do treinamento
        self.iterations += 1


class PrintLRScheduler(Callback):
    def on_epoch_end(self, net, **kwargs):
        lr = net.optimizer_.param_groups[0]['lr']
        log.info(f"Epoch {len(net.history)} - Learning Rate: {lr:.6f}")


class DisableSkipConnections(Callback):
    def __init__(self):
        self.is_first_time = True

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        log.info(f"Disabling skip connections.")

        if self.is_first_time:
            self.is_first_time = False

        net.module_.use_skip_connections = False


class TwoPhaseTraining(Callback):
    def __init__(self, nmb_of_epochs=500, step1_perc=0.5, warmup_perc=0.25, finetune_lr=1e-5):
        self.step1_epochs = nmb_of_epochs * step1_perc
        self.warmup_epochs = nmb_of_epochs * warmup_perc
        self.finetune_lr = finetune_lr
        self.is_first_time = True

    def on_epoch_begin(self, net, **kwargs):
        # In skorch, len(net.history) gives the current 0-indexed epoch number
        # because the history only appends after an epoch finishes.
        current_epoch = len(net.history)

        # ==========================================
        # TRIGGER STEP 1: Initialization 
        # ==========================================
        if self.is_first_time:
            self.is_first_time = False

            log.info(f"[Step 1] Epoch {current_epoch}: Skips OFF, training pure bottleneck...")
            
            # 1. Force skip connections OFF
            net.module_.use_skip_connections = False
            
            # 2. Ensure all layers are fully trainable
            for param in net.module_.parameters():
                param.requires_grad = True
                
        elif current_epoch == self.step1_epochs:
            log.info(f"\n[Step 2 - Phase A] Epoch {current_epoch}: Skips ON, Freezing Encoder/Latent...")
            
            # 1. Activate skip connections
            net.module_.use_skip_connections = True
            
            # 2. Freeze Encoder and Latent blocks
            for param in net.module_.encoder.parameters():
                param.requires_grad = False
            for param in net.module_.latent.parameters():
                param.requires_grad = False
                
            # Ensure Decoder is trainable
            for param in net.module_.decoder.parameters():
                param.requires_grad = True
                
            # 3. CRITICAL: Re-initialize the optimizer. 
            # Skorch automatically filters for requires_grad=True during this call,
            # and it flushes all historical momentum from AdamW.
            net.initialize_optimizer()

        # ==========================================
        # TRIGGER PHASE B: Full Fine-tuning
        # ==========================================
        elif current_epoch == self.step1_epochs + self.warmup_epochs:
            log.info(f"\n[Step 2 - Phase B] Epoch {current_epoch}: Unfreezing all, lowering LR...")
            
            # 1. Unfreeze all layers
            for param in net.module_.parameters():
                param.requires_grad = True
                
            # 2. Lower the global learning rate
            net.lr = self.finetune_lr
            
            # 3. Re-initialize the optimizer to pick up the un-frozen layers 
            # and apply the new learning rate cleanly.
            net.initialize_optimizer()


class MLModelsUtils:

    def get_skorch_disable_skip_connections_callback():
        return DisableSkipConnections()

    def get_skorch_two_step_training_callback(nmb_of_epochs=500, step1_perc=0.5, warmup_perc=0.25, finetune_lr=1e-5):
        return TwoPhaseTraining(
            nmb_of_epochs=nmb_of_epochs, 
            step1_perc=step1_perc, 
            warmup_perc=warmup_perc, 
            finetune_lr=finetune_lr
        )

    def get_skorch_monitor_gradient_clipping_callback(clip_value=None, print_every=5):
        return MonitoredGradClip(clip_value=clip_value, print_every=print_every)

    def get_skorch_gradient_clipping_callback(clip_value=None):
        return CustomGradientClipping(clip_value=clip_value)

    def get_skorch_loadInitState_callback(checkpoint_cb):
        return LoadInitState(checkpoint_cb)

    def get_skorch_checkpoint_callback(fn_prefix='', dirname='', load_best=True):
        return Checkpoint(
                            monitor='valid_loss_best',
                            f_params='params.pt',
                            f_optimizer='optimizer.pt',
                            f_criterion='criterion.pt',
                            f_history='history.json',
                            fn_prefix=fn_prefix,
                            dirname=dirname,
                            event_name='event_cp',
                            load_best=load_best
        )

    def get_skorch_early_stopping_callback(patience=20):
        return EarlyStopping(
                            monitor='valid_loss', # explicitly for legibility
                            patience=patience
        )

    def get_skorch_lr_scheduler_lambda_callback(decay=0.0002):
        lr_lambda = lambda epoch: 1 / (1 + decay * epoch)
        return LRScheduler(policy=LambdaLR, lr_lambda=lr_lambda), "decay" 

    def get_skorch_lr_scheduler_exponential_callback(gamma=0.95):
        return LRScheduler(policy=ExponentialLR, gamma=gamma), "exponential"

    def get_skorch_lr_scheduler_cosine_annealing_warm_restarts_callback(T_0=10, T_mult=3, eta_min=1e-6):
        return LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult, eta_min=eta_min), "cosine_annealing_wr"

    def get_skorch_lr_scheduler_cosine_annealing_callback(T_max=150):
        return LRScheduler(policy=CosineAnnealingLR, T_max=T_max), "cosine_annealing"

    def get_skorch_print_lr_scheduler_callback():
        return PrintLRScheduler()









