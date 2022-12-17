
class Config():
    def __init__(self):
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [30, 50]#scheduler = optim.lr_scheduler.MultiStepLR
        self.input_size = 256
        self.backbone = 'resnet18'
        self.pretrained = True
        self.criterion_reg = 'l1'
        self.num_lms = 98
        self.use_gpu = True
        self.gpu_id = 0
