class RunConfig:
    def __init__(self, args):
        self.batch_size = 32
        self.train_epochs = 20
        self.calibration_epochs = 100
        self.learning_rate = 0.0001
        self.training_size = 0.9
        self.calibration_size = 0.0333
        self.train_subjects = [f"p{pid:02}" for pid in range(00, 14)]
        self.test_subjects = ["p14"]
        self.channel_selections = [192, 3, 9, 12, 20, 22, 35, 192]
        self.model_runs = 10
        self.data_dir = "data"
        self.saves_dir = "models/saves"
        self.images = args.images
        self.model = args.model
        self.data = args.data
        self.lc_hc = args.lc_hc
        self.model_id = args.model_id
        self.run = args.run
