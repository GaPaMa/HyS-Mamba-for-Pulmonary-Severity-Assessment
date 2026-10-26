"""Run the attached author trainer with explicit seeding and VSS verification."""
from accelerate.utils import set_seed
from lung_severity_assesmnt.configs import parse_configs
from lung_severity_assesmnt.trainer import AgeEstimationTrainer
from lung_severity_assesmnt.utils.reproducibility import ensure_state_space


class HySMambaTrainer(AgeEstimationTrainer):
    def setup_model(self):
        super().setup_model()
        ensure_state_space(self.model)


if __name__ == "__main__":
    configs = parse_configs(training=True)
    set_seed(configs.seed if configs.seed is not None else 42)
    trainer = HySMambaTrainer(configs)
    trainer()
