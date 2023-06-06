class RunConfig:
    def __init__(self, args):
        self.batch_size = 32
        self.train_epochs = 20
        self.calibration_epochs = 100
        self.learning_rate = 0.0001
        self.calibration_size = 0.0333
        self.data_dir = "data"
        self.saves_dir = "models/saves"
        self.train_subjects = [f"p{pid:02}" for pid in range(00, 14)]
        self.test_subjects = ["p14"]
        self.images = args.images
        self.model = args.model
        self.data_type = args.data_type
        self.model_id = args.model_id
