from deepspeed.utils import RepeatingLoader


def test_repeating_loader():
    loader = [1, 2, 3]
    loader = RepeatingLoader(loader)

    for idx in range(50):
        assert next(loader) == 1
        assert next(loader) == 2
        assert next(loader) == 3
