import deepspeed
from unit.common import DistributedTest
from unit.simple_model import Curriculum_SimpleModel, random_dataloader


class TestCurriculumScheduler(DistributedTest):
    world_size = 2

    def test_fixed_discrete(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
            "curriculum_learning": {
                "enabled": True,
                "curriculum_type": "seqlen",
                "min_difficulty": 1,
                "max_difficulty": 5,
                "schedule_type": "fixed_discrete",
                "schedule_config": {
                    "difficulty": [1,
                                   2,
                                   3,
                                   4,
                                   5],
                    "max_step": [2,
                                 4,
                                 6,
                                 8]
                }
            }
        }
        hidden_dim = 10
        ground_truths = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}

        model = Curriculum_SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss, seqlen = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            true_seqlen = 5
            if n + 1 in ground_truths:
                true_seqlen = ground_truths[n + 1]
            assert seqlen == true_seqlen, f"Incorrect curriculum schedule"

    def test_fixed_linear(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16
            },
            "curriculum_learning": {
                "enabled": True,
                "curriculum_type": "seqlen",
                "min_difficulty": 2,
                "max_difficulty": 10,
                "schedule_type": "fixed_linear",
                "schedule_config": {
                    "total_curriculum_step": 8,
                    "difficulty_step": 2
                }
            }
        }
        hidden_dim = 10
        ground_truths = {1: 2, 2: 4, 3: 4, 4: 6, 5: 6, 6: 8, 7: 8, 8: 10, 9: 10, 10: 10}

        model = Curriculum_SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=20,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss, seqlen = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            if n + 1 in ground_truths:
                true_seqlen = ground_truths[n + 1]
                assert seqlen == true_seqlen, f"Incorrect curriculum schedule"
