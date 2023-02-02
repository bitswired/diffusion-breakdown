from diffusion_breakdown.data import get_moons_data_pipe


class TestData:
    def test_moons(self):
        data_pipeline = get_moons_data_pipe(10)
        for x in data_pipeline:
            assert x.shape == (10, 2)
            break
