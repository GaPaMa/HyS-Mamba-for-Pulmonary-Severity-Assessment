"""Run the attached author trainer with explicit seeding and VSS verification."""
from accelerate.utils import set_seed
from age_estimation.configs import parse_configs
from age_estimation.trainer import AgeEstimationTrainer
from age_estimation.utils.reproducibility import ensure_state_space


class HySMambaTrainer(AgeEstimationTrainer):
    def setup_model(self):
        super().setup_model()
        ensure_state_space(self.model)


if __name__ == "__main__":
    configs = parse_configs(training=True)
    set_seed(configs.seed if configs.seed is not None else 42)
    trainer = HySMambaTrainer(configs)
    trainer()
